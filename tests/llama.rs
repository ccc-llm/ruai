//! Llama-family building blocks (RMSNorm, RoPE, SwiGLU, GQA) and the
//! `Llama` model: KV cache, Hugging Face layout, training, int8.
//! Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::functional as F;
use ruai::generation::{argmax, Sampler};
use ruai::nn::llama::{Llama, LlamaConfig, RopeScaling};
use ruai::nn::{KvCache, Module};
use ruai::optim::{Adam, Optimizer};
use ruai::{no_grad, rand, serialize, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// GQA: 4 query heads sharing 2 key/value heads.
fn tiny(tied: bool) -> Llama {
    let mut c = LlamaConfig::new(41, 32, 2, 4, 2, 16);
    c.tie_word_embeddings = tied;
    Llama::new(c, dev())
}

#[test]
fn rms_norm_values() {
    let x = Tensor::from_vec(vec![1., 2., 3., 4., -2., 0., 2., 0.], &[2, 4], dev());
    let w = Tensor::from_vec(vec![1., 2., 1., 0.5], &[4], dev());
    let y = F::rms_norm(&x, Some(&w), 0.0).to_vec();
    let r0 = (30f32 / 4.0).sqrt();
    let r1 = 2f32.sqrt();
    let want = [1. / r0, 4. / r0, 3. / r0, 2. / r0, -2. / r1, 0., 2. / r1, 0.];
    assert!(max_abs_diff(&y, &want) < 1e-5, "{y:?}");
}

#[test]
fn rope_is_a_relative_rotation() {
    let _l = lock();
    rand::manual_seed(3);
    let d = 8;
    let inv = F::rope_inv_freq(d, 10000.0);
    let q = Tensor::randn(&[1, d], dev());
    let k = Tensor::randn(&[1, d], dev());
    let at = |x: &Tensor, pos: usize| {
        let (c, s) = F::rope_cos_sin(&inv, pos, 1, dev());
        F::apply_rope(x, &c, &s)
    };
    let dot = |a: &Tensor, b: &Tensor| (a * b).sum_all().item();
    // norms are preserved
    let (n0, n1) = (dot(&q, &q), dot(&at(&q, 7), &at(&q, 7)));
    assert!((n0 - n1).abs() < 1e-4 * n0.max(1.0));
    // <rope(q, m), rope(k, n)> depends only on m - n
    let a = dot(&at(&q, 5), &at(&k, 2));
    let b = dot(&at(&q, 13), &at(&k, 10));
    assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    // position 0 is the identity
    assert!(max_abs_diff(&at(&q, 0).to_vec(), &q.to_vec()) < 1e-6);
    // table layout: frequencies repeated for both halves
    let (c, _) = F::rope_cos_sin(&inv, 3, 1, dev());
    let c = c.to_vec();
    for i in 0..d / 2 {
        assert_eq!(c[i], c[i + d / 2]);
        assert!((c[i] - (3.0 * inv[i]).cos()).abs() < 1e-6);
    }
}

#[test]
fn repeat_kv_layout() {
    // [1, 2 heads, 1, 2] -> each head repeated twice in a row
    let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[1, 2, 1, 2], dev());
    let y = F::repeat_kv(&x, 2);
    assert_eq!(y.shape(), &[1, 4, 1, 2]);
    assert_eq!(y.to_vec(), vec![1., 2., 1., 2., 3., 4., 3., 4.]);
}

#[test]
fn llama3_rope_scaling() {
    let mut c = LlamaConfig::new(10, 128, 1, 2, 1, 32);
    c.rope_theta = 500000.0;
    let base = c.rope_inv_freq();
    c.rope_scaling = Some(RopeScaling {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192,
    });
    let scaled = c.rope_inv_freq();
    let two_pi = 2.0 * std::f32::consts::PI;
    for (f, s) in base.iter().zip(&scaled) {
        let wavelen = two_pi / f;
        if wavelen < 8192.0 / 4.0 {
            assert_eq!(f, s, "high frequencies are kept");
        } else if wavelen > 8192.0 {
            assert!((s - f / 8.0).abs() <= 1e-7 * f, "low frequencies are divided by the factor");
        } else {
            assert!(*s < *f && *s > f / 8.0, "medium frequencies are interpolated");
        }
    }
    assert_eq!(base[0], scaled[0]);
    assert!(scaled.last().unwrap() < base.last().unwrap());
}

#[test]
fn config_from_hf_json() {
    // TinyLlama-1.1B's config.json (abridged)
    let json = r#"{
        "architectures": ["LlamaForCausalLM"], "hidden_act": "silu", "hidden_size": 2048,
        "intermediate_size": 5632, "max_position_embeddings": 2048, "model_type": "llama",
        "num_attention_heads": 32, "num_hidden_layers": 22, "num_key_value_heads": 4,
        "rms_norm_eps": 1e-05, "rope_scaling": null, "tie_word_embeddings": false,
        "torch_dtype": "bfloat16", "vocab_size": 32000
    }"#;
    let c = LlamaConfig::from_json(json).unwrap();
    assert_eq!((c.hidden_size, c.num_key_value_heads, c.head_dim, c.vocab_size), (2048, 4, 64, 32000));
    assert_eq!(c.rope_theta, 10000.0);
    assert!(c.rope_scaling.is_none() && !c.tie_word_embeddings);
    assert_eq!(LlamaConfig::from_json(&c.to_json()).unwrap(), c);

    let llama3 = r#"{"hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 64,
        "num_attention_heads": 4, "num_hidden_layers": 1, "vocab_size": 9, "rope_theta": 500000.0,
        "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                         "original_max_position_embeddings": 8192, "rope_type": "llama3"}}"#;
    let c = LlamaConfig::from_json(llama3).unwrap();
    assert_eq!(c.rope_scaling.unwrap().factor, 32.0);
    assert_eq!(c.num_key_value_heads, 4, "defaults to num_attention_heads");
    assert_eq!(LlamaConfig::from_json(&c.to_json()).unwrap(), c);
    assert!(LlamaConfig::from_json(&llama3.replace("llama3", "yarn")).is_err());
}

fn check_cache(model: &Llama, ids: &[usize], chunks: &[usize]) {
    let n = ids.len();
    let full = model.forward_tokens(&Tensor::from_indices(ids, &[1, n], dev()));
    let v = full.dim(-1);
    let mut cache = KvCache::default();
    let mut pos = 0;
    for &c in chunks {
        let out = model.forward_cached(&Tensor::from_indices(&ids[pos..pos + c], &[1, c], dev()), &mut cache);
        assert_eq!(cache.len(), pos + c);
        let want = full.narrow(1, pos, c).reshape(&[c * v]).to_vec();
        assert!(max_abs_diff(&out.reshape(&[c * v]).to_vec(), &want) < 1e-4, "positions {pos}..{}", pos + c);
        pos += c;
    }
}

#[test]
fn kv_cache_matches_full_forward() {
    let _l = lock();
    rand::manual_seed(1);
    let ids: Vec<usize> = (0..12).map(|i| (i * 7 + 3) % 41).collect();
    no_grad(|| {
        for tied in [false, true] {
            let m = tiny(tied);
            check_cache(&m, &ids, &[1; 12]);
            check_cache(&m, &ids, &[5, 1, 4, 2]);
            // the cache keeps only the 2 key/value heads
            let mut cache = KvCache::default();
            m.forward_cached(&Tensor::from_indices(&ids[..3], &[1, 3], dev()), &mut cache);
            assert_eq!(cache.layers[0].k.as_ref().unwrap().shape(), &[1, 2, 3, 8]);
        }
    });
}

#[test]
fn causal_and_batched() {
    let _l = lock();
    rand::manual_seed(2);
    let m = tiny(false);
    no_grad(|| {
        let a: Vec<usize> = vec![1, 2, 3, 4, 5];
        let mut b = a.clone();
        b[4] = 9;
        let la = m.forward_tokens(&Tensor::from_indices(&a, &[1, 5], dev())).to_vec();
        let lb = m.forward_tokens(&Tensor::from_indices(&b, &[1, 5], dev())).to_vec();
        let v = 41;
        assert!(max_abs_diff(&la[..4 * v], &lb[..4 * v]) < 1e-5, "earlier positions must not see the last token");
        // a batch of two sequences equals two separate passes
        let both: Vec<usize> = a.iter().chain(&b).copied().collect();
        let lab = m.forward_tokens(&Tensor::from_indices(&both, &[2, 5], dev())).to_vec();
        assert!(max_abs_diff(&lab[..5 * v], &la) < 1e-5);
        assert!(max_abs_diff(&lab[5 * v..], &lb) < 1e-5);
    });
}

#[test]
fn generate_with_cache_equals_without() {
    let _l = lock();
    rand::manual_seed(4);
    let m = tiny(true);
    let prompt = [3usize, 1, 4];
    // 20 new tokens > max_position_embeddings (16): exercises the sliding window
    let got = m.generate(&prompt, 20, &mut Sampler::greedy());
    let mut want = prompt.to_vec();
    no_grad(|| {
        for _ in 0..20 {
            let ctx = &want[want.len().saturating_sub(16)..];
            let l = m.forward_tokens(&Tensor::from_indices(ctx, &[1, ctx.len()], dev()));
            let v = l.dim(-1);
            want.push(argmax(&l.narrow(1, ctx.len() - 1, 1).reshape(&[v]).to_vec()));
        }
    });
    // after the window slides, the cached path re-encodes and then continues
    // incrementally, while this reference re-encodes every step; both agree
    // until the first slide
    assert_eq!(got[..16], want[..16]);
    assert_eq!(got.len(), 23);
    // stopping early
    let mut count = 0;
    let short = m.generate_with(&prompt, 10, &mut Sampler::greedy(), |_| {
        count += 1;
        count < 4
    });
    assert_eq!(short.len(), 7);
}

#[test]
fn hugging_face_layout_roundtrip() {
    let _l = lock();
    rand::manual_seed(5);
    let dir = std::env::temp_dir().join(format!("ruai_llama_{}", std::process::id()));
    for tied in [false, true] {
        let m = tiny(tied);
        m.save_pretrained(&dir).unwrap();

        // the file is in HF layout: Linear weights are [out, in]
        let (t, meta) = serialize::load_safetensors(dir.join("model.safetensors")).unwrap();
        assert_eq!(meta.get("format").map(String::as_str), Some("pt"));
        let get = |n: &str| t.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
        assert_eq!(get("model.layers.0.self_attn.q_proj.weight").unwrap().shape(), &[32, 32]);
        assert_eq!(get("model.layers.1.self_attn.k_proj.weight").unwrap().shape(), &[16, 32]);
        let gate = get("model.layers.0.mlp.gate_proj.weight").unwrap();
        let inter = m.config.intermediate_size;
        assert_eq!(gate.shape(), &[inter, 32]);
        assert_eq!(get("model.embed_tokens.weight").unwrap().shape(), &[41, 32]);
        assert_eq!(get("model.norm.weight").unwrap().shape(), &[32]);
        assert_eq!(get("lm_head.weight").map(|h| h.shape().to_vec()), (!tied).then(|| vec![41, 32]));
        // element (o, i) of the HF tensor is element (i, o) of ours
        let ours = m.named_parameters().into_iter().find(|(n, _)| n == "model.layers.0.mlp.gate_proj.weight").unwrap().1;
        let (hv, ov) = (gate.to_vec(), ours.to_vec());
        assert_eq!(hv[5 * 32 + 7], ov[7 * inter + 5]);

        let loaded = Llama::from_pretrained(&dir, dev()).unwrap();
        assert_eq!(loaded.config, m.config);
        let ids = Tensor::from_indices(&[1, 5, 9, 2], &[1, 4], dev());
        no_grad(|| {
            let a = m.forward_tokens(&ids).to_vec();
            let b = loaded.forward_tokens(&ids).to_vec();
            assert!(max_abs_diff(&a, &b) < 1e-6);
        });
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn learns_a_sequence() {
    let _l = lock();
    rand::manual_seed(6);
    let model = Llama::new(LlamaConfig::new(5, 16, 1, 2, 1, 8), dev());
    let seq = [0usize, 1, 2, 3, 4, 0, 1, 2, 3];
    let inp = Tensor::from_indices(&seq[..8], &[1, 8], dev());
    let mut opt = Adam::new(model.parameters(), 0.01);
    for _ in 0..150 {
        opt.zero_grad();
        let logits = model.forward_tokens(&inp).reshape(&[8, 5]);
        F::cross_entropy(&logits, &seq[1..], 0.0).backward();
        opt.step();
    }
    assert_eq!(model.generate(&[0], 7, &mut Sampler::greedy()), vec![0, 1, 2, 3, 4, 0, 1, 2]);
}

#[test]
fn int8_is_close() {
    let _l = lock();
    rand::manual_seed(7);
    for tied in [false, true] {
        let mut m = tiny(tied);
        let ids = Tensor::from_indices(&[1, 5, 9, 2, 7], &[1, 5], dev());
        let before = no_grad(|| m.forward_tokens(&ids).to_vec());
        m.quantize_int8();
        let after = no_grad(|| m.forward_tokens(&ids).to_vec());
        let scale = before.iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(max_abs_diff(&before, &after) < 0.05 * scale.max(0.1), "tied={tied}");
        assert!(m.named_buffers().iter().any(|(n, _)| n == "lm_head.weight_int8"));
    }
}

#[test]
fn autocast_gradients_match_f32() {
    let _l = lock();
    rand::manual_seed(8);
    let m = tiny(false);
    let ids = Tensor::from_indices(&[1, 5, 9, 2, 7, 3], &[1, 6], dev());
    let tgt = [5usize, 9, 2, 7, 3, 1];
    let grads = |amp: Option<ruai::DType>| {
        m.zero_grad();
        let f = || F::cross_entropy(&m.forward_tokens(&ids).reshape(&[6, 41]), &tgt, 0.0);
        let loss = match amp {
            Some(d) => ruai::autocast(d, f),
            None => f(),
        };
        loss.backward();
        m.parameters().iter().flat_map(|p| p.grad().unwrap().to_vec()).collect::<Vec<f32>>()
    };
    let g32 = grads(None);
    let g16 = grads(Some(ruai::DType::F16));
    let norm = g32.iter().map(|x| x * x).sum::<f32>().sqrt();
    let err = g32.iter().zip(&g16).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    assert!(err < 0.02 * norm, "relative gradient error {}", err / norm);
}
