//! KV-cache decoding, GPT-2 loading, int8 quantization, the BPE tokenizer
//! and sampling. Set `RUSTNN_DEVICE=mps` to run on the GPU.
//!
//! The golden tests against the real GPT-2 weights run only when
//! `data/gpt2/{model.safetensors,vocab.json,merges.txt}` exist
//! (see `scripts/download_gpt2.sh`).

use ruai::generation::{argmax, Sampler};
use ruai::nn::gpt2::{GPT2Config, GPT2};
use ruai::nn::{self, KvCache, Linear, Module};
use ruai::tokenizer::{pre_tokenize, Gpt2Tokenizer};
use ruai::{no_grad, rand, serialize, Device, Tensor};
use std::path::Path;

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

fn tiny_gpt2() -> GPT2 {
    GPT2::new(GPT2Config { vocab_size: 37, n_positions: 16, n_embd: 32, n_layer: 2, n_head: 4 }, dev())
}

/// Logits for each position computed one token at a time with the cache
/// must equal a single full forward pass.
fn check_cache(full: &Tensor, step: impl Fn(&Tensor, &mut KvCache) -> Tensor, ids: &[usize], chunks: &[usize]) {
    let v = full.dim(-1);
    let mut cache = KvCache::default();
    let mut pos = 0;
    for &c in chunks {
        let out = step(&Tensor::from_indices(&ids[pos..pos + c], &[1, c], dev()), &mut cache);
        assert_eq!(cache.len(), pos + c);
        let want = full.narrow(1, pos, c).reshape(&[c * v]).to_vec();
        let got = out.reshape(&[c * v]).to_vec();
        assert!(max_abs_diff(&got, &want) < 1e-4, "cached logits differ at positions {pos}..{}", pos + c);
        pos += c;
    }
}

#[test]
fn kv_cache_matches_full_forward() {
    let _l = lock();
    rand::manual_seed(1);
    let ids: Vec<usize> = (0..10).map(|i| (i * 7 + 3) % 37).collect();
    no_grad(|| {
        let g2 = tiny_gpt2();
        let full = g2.forward_tokens(&Tensor::from_indices(&ids, &[1, 10], dev()));
        check_cache(&full, |t, c| g2.forward_cached(t, c), &ids, &[1; 10]);
        check_cache(&full, |t, c| g2.forward_cached(t, c), &ids, &[4, 1, 3, 2]); // prefill + decode

        let gpt = nn::GPT::new(nn::GPTConfig { vocab_size: 37, max_len: 16, dim: 16, n_heads: 2, n_layers: 2, dropout: 0.0 }, dev());
        gpt.eval();
        let full = gpt.forward_ids(&ids, 1, 10);
        check_cache(&full, |t, c| gpt.forward_cached(t, c), &ids, &[3, 1, 1, 5]);
        // cached generation == recomputing everything (also past max_len: window slides)
        assert_eq!(gpt.generate(&ids[..3], 20, &mut Sampler::greedy()), gpt.generate_uncached(&ids[..3], 20));
    });
}

#[test]
fn gpt2_uses_hugging_face_names_and_loads_checkpoints() {
    let _l = lock();
    rand::manual_seed(2);
    // n_embd = 64 * n_head, so from_pretrained infers the head count (as for real GPT-2 sizes)
    let cfg = GPT2Config { vocab_size: 50, n_positions: 12, n_embd: 128, n_layer: 2, n_head: 2 };
    let a = GPT2::new(cfg, dev());
    let params = a.named_parameters();
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    for want in ["wte.weight", "wpe.weight", "h.0.ln_1.weight", "h.0.attn.c_attn.weight", "h.0.attn.c_attn.bias", "h.1.mlp.c_proj.bias", "ln_f.bias"] {
        assert!(names.contains(&want), "missing parameter {want}");
    }
    let c_attn = &params.iter().find(|(n, _)| n == "h.0.attn.c_attn.weight").unwrap().1;
    assert_eq!(c_attn.shape(), &[128, 384], "Conv1D layout [in, out]");

    // an HF-style file: "transformer." prefix, causal-mask buffers and a tied lm_head
    let mut tensors: Vec<(String, Tensor)> =
        a.state_dict().into_iter().map(|(n, t)| (format!("transformer.{n}"), t)).collect();
    tensors.push(("transformer.h.0.attn.bias".into(), Tensor::ones(&[1, 1, 12, 12], Device::Cpu)));
    tensors.push(("transformer.h.1.attn.masked_bias".into(), Tensor::scalar(-1e4, Device::Cpu)));
    tensors.push(("lm_head.weight".into(), a.wte.weight.clone()));
    let dir = std::env::temp_dir().join(format!("ruai_gpt2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    serialize::save_safetensors(dir.join("model.safetensors"), &tensors, None).unwrap();
    let b = GPT2::from_pretrained(&dir, dev()).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(b.config, cfg);
    let ids = Tensor::from_indices(&[1, 5, 9, 2, 49], &[1, 5], dev());
    no_grad(|| assert_eq!(a.forward_tokens(&ids).to_vec(), b.forward_tokens(&ids).to_vec()));
}

#[test]
fn int8_quantization_is_accurate() {
    let _l = lock();
    rand::manual_seed(4);
    let mut lin = Linear::new(64, 48, dev());
    let x = Tensor::randn(&[3, 64], dev());
    let want = no_grad(|| lin.forward(&x)).to_vec();
    lin.quantize_int8();
    assert!(lin.is_quantized());
    let names: Vec<String> = lin.state_dict().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["bias", "weight_int8", "weight_scale"]);
    let got = no_grad(|| lin.forward(&x)).to_vec();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(max_abs_diff(&got, &want) / scale < 0.02, "int8 linear error {}", max_abs_diff(&got, &want) / scale);

    // whole GPT-2: logits stay close, greedy next tokens mostly agree
    let mut g = GPT2::new(GPT2Config { vocab_size: 64, n_positions: 16, n_embd: 64, n_layer: 2, n_head: 2 }, dev());
    let ids: Vec<usize> = (0..12).map(|i| (i * 5 + 1) % 64).collect();
    let t = Tensor::from_indices(&ids, &[1, 12], dev());
    let reference = no_grad(|| g.forward_tokens(&t)).to_vec();
    g.quantize_int8();
    let quant = no_grad(|| g.forward_tokens(&t)).to_vec();
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(max_abs_diff(&quant, &reference) / scale < 0.05);
    let agree = reference.chunks(64).zip(quant.chunks(64)).filter(|(a, b)| argmax(a) == argmax(b)).count();
    assert!(agree >= 10, "only {agree}/12 positions agree");
}

/// A self-contained BPE vocabulary: every byte, plus a few merges.
fn synthetic_tokenizer() -> Gpt2Tokenizer {
    // reproduce GPT-2's byte -> char table to build the base vocab
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = bytes.clone();
    let mut extra = 0;
    for b in 0..256 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let mut vocab: Vec<String> = chars.iter().map(|&c| char::from_u32(c).unwrap().to_string()).collect();
    let merges = ["h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "r l", "Ġwo rl", "Ġworl d"];
    for m in merges {
        vocab.push(m.replace(' ', ""));
    }
    let json = format!(
        "{{{}}}",
        vocab.iter().enumerate().map(|(i, t)| format!("{}:{i}", serde_json::to_string(t).unwrap())).collect::<Vec<_>>().join(",")
    );
    Gpt2Tokenizer::from_strings(&json, &format!("#version: 0.2\n{}\n", merges.join("\n"))).unwrap()
}

#[test]
fn bpe_tokenizer_merges_and_roundtrips() {
    let tok = synthetic_tokenizer();
    assert_eq!(tok.vocab_size(), 256 + 9);
    // "hello" -> one merged token, " world" -> one merged token (Ġ is the space byte)
    let ids = tok.encode("hello world");
    assert_eq!(ids, vec![256 + 3, 256 + 8]);
    assert_eq!(tok.decode(&ids), "hello world");
    // anything round-trips byte-exactly, merged or not
    for s in ["", "hello\n\n  world  ", "héllo wörld 你好 🎉", "it's 2024!! don't", "tabs\tand\r\nnewlines"] {
        assert_eq!(tok.decode(&tok.encode(s)), s, "{s:?}");
    }
}

#[test]
fn pre_tokenizer_matches_gpt2_regex() {
    let cases: [(&str, &[&str]); 8] = [
        ("Hello world", &["Hello", " world"]),
        ("  hi", &[" ", " hi"]),
        ("it's 2024!!", &["it", "'s", " 2024", "!!"]),
        ("a  \n b", &["a", "  \n", " b"]),
        ("hi  ", &["hi", "  "]),
        ("I'll we've", &["I", "'ll", " we", "'ve"]),
        ("你好 世界", &["你好", " 世界"]),
        ("x=1+2", &["x", "=", "1", "+", "2"]),
    ];
    for (text, want) in cases {
        assert_eq!(pre_tokenize(text), want, "{text:?}");
    }
}

#[test]
fn sampler_behaviour() {
    let logits = [1.0, 3.0, 2.0, -1.0];
    assert_eq!(Sampler::greedy().sample(&logits), 1);
    assert_eq!(Sampler::new(1.0, 1, 1.0, 7).sample(&logits), 1, "top_k = 1 is greedy");
    assert_eq!(Sampler::new(1.0, 0, 0.01, 7).sample(&logits), 1, "tiny top_p is greedy");
    // top_k = 2 never picks outside the two best
    let mut s = Sampler::new(1.0, 2, 1.0, 3);
    assert!((0..500).all(|_| matches!(s.sample(&logits), 1 | 2)));
    // frequencies follow softmax at temperature 1
    let mut s = Sampler::new(1.0, 0, 1.0, 11);
    let mut counts = [0usize; 4];
    for _ in 0..20000 {
        counts[s.sample(&logits)] += 1;
    }
    let z: f32 = logits.iter().map(|l| l.exp()).sum();
    for (c, l) in counts.iter().zip(logits) {
        let p = l.exp() / z;
        assert!((*c as f32 / 20000.0 - p).abs() < 0.015, "{counts:?}");
    }
    // seeded: reproducible
    let run = |seed| (0..20).map(|_| 0).scan(Sampler::new(0.8, 0, 0.95, seed), |s, _| Some(s.sample(&logits))).collect::<Vec<_>>();
    assert_eq!(run(5), run(5));
}

// ---------------------------------------------------------------- real GPT-2

const GPT2_DIR: &str = "data/gpt2";

fn gpt2_files() -> bool {
    let ok = ["model.safetensors", "vocab.json", "merges.txt"].iter().all(|f| Path::new(GPT2_DIR).join(f).exists());
    if !ok {
        eprintln!("skipping: GPT-2 files not found in {GPT2_DIR} (run scripts/download_gpt2.sh)");
    }
    ok
}

#[test]
fn gpt2_tokenizer_golden() {
    if !gpt2_files() {
        return;
    }
    let tok = Gpt2Tokenizer::from_dir(GPT2_DIR).unwrap();
    assert_eq!(tok.vocab_size(), 50257);
    assert_eq!(tok.encode("The dog"), vec![464, 3290]);
    assert_eq!(tok.encode("Hello world"), vec![15496, 995]);
    let s = "Rust 是一種系統程式語言 — it's fast!\n\n  Indented.";
    assert_eq!(tok.decode(&tok.encode(s)), s);
}

/// Greedy continuation of "The dog" from Hugging Face transformers' own
/// GPT-2 integration test (`test_lm_generate_gpt2`).
const HF_GREEDY_THE_DOG: [usize; 20] =
    [464, 3290, 373, 1043, 287, 257, 2214, 1474, 262, 16246, 286, 2688, 290, 2688, 27262, 13, 198, 198, 464, 3290];

#[test]
fn gpt2_greedy_matches_hugging_face() {
    if !gpt2_files() {
        return;
    }
    let model = GPT2::from_pretrained(GPT2_DIR, dev()).unwrap();
    assert_eq!(model.config, GPT2Config::gpt2());
    let out = model.generate(&[464, 3290], 18, &mut Sampler::greedy());
    assert_eq!(out, HF_GREEDY_THE_DOG);
    // int8: logits within a few percent. Greedy text may still diverge
    // early: after "The dog", f32 prefers " was" over "'s" by only 0.33
    // logits, less than int8's typical error (~0.5)
    let mut q = GPT2::from_pretrained(GPT2_DIR, dev()).unwrap();
    q.quantize_int8();
    let t = Tensor::from_indices(&[464, 3290], &[1, 2], dev());
    let v = 50257;
    let (lf, lq) = no_grad(|| (model.forward_tokens(&t).to_vec(), q.forward_tokens(&t).to_vec()));
    let scale = lf.iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!(max_abs_diff(&lf, &lq) / scale < 0.03, "int8 logit error {}", max_abs_diff(&lf, &lq) / scale);
    let (rf, rq) = (&lf[v..], &lq[v..]);
    let best = argmax(rf);
    let second = (0..v).filter(|&i| i != best).max_by(|&a, &b| rf[a].total_cmp(&rf[b])).unwrap();
    assert!([best, second].contains(&argmax(rq)), "int8 picks a token outside f32's top 2");
    assert_eq!(q.generate(&[464, 3290], 18, &mut Sampler::greedy()).len(), 20);
}
