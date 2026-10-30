//! Weight-only quantization: int4 packing, accuracy, models, LoRA on int4.
//! Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::backend::{backend_for, reference::Unfused, Backend};
use ruai::functional as F;
use ruai::generation::argmax;
use ruai::nn::gpt2::{GPT2Config, GPT2};
use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::nn::{Linear, Module};
use ruai::optim::{Adam, Optimizer};
use ruai::{no_grad, rand, Device, Tensor};

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

#[test]
fn int4_packing_matches_an_independent_reference() {
    let _l = lock();
    rand::manual_seed(0);
    let (k, n, group) = (8, 3, 4);
    let mut lin = Linear::with_bias(k, n, false, dev());
    let w = lin.weight.to_vec();
    lin.quantize_int4(group);
    let names: Vec<String> = lin.state_dict().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["weight_int4", "weight_int4_scale"]);
    assert_eq!(lin.dims(), (k, n));
    // expected dequantized weights, computed directly from the definition
    let mut want = vec![0f32; k * n];
    for j in 0..n {
        for g in 0..k / group {
            let rows = g * group..(g + 1) * group;
            let s = rows.clone().map(|l| w[l * n + j].abs()).fold(0f32, f32::max) / 7.0;
            for l in rows {
                want[l * n + j] = (w[l * n + j] / s).round().clamp(-8.0, 7.0) * s;
            }
        }
    }
    // the identity matrix reads the weights back through matmul_int4
    let eye = Tensor::from_vec((0..k * k).map(|i| if i / k == i % k { 1.0 } else { 0.0 }).collect(), &[k, k], dev());
    let got = no_grad(|| lin.forward(&eye)).to_vec();
    assert!(max_abs_diff(&got, &want) < 1e-6, "{got:?}\nvs {want:?}");
    // fused kernel == the backend's default (host) dequantization + GEMM
    let sd = lin.state_dict();
    let (q, s) = (sd[0].1.storage(), sd[1].1.storage());
    let x = Tensor::randn(&[2, k], dev());
    let be = backend_for(dev());
    let reference = Unfused(be).matmul_int4(&x.storage(), &q, &s, 2, k, n, group).to_vec();
    assert!(max_abs_diff(&be.matmul_int4(&x.storage(), &q, &s, 2, k, n, group).to_vec(), &reference) < 1e-5);
}

#[test]
fn int4_linear_is_close() {
    let _l = lock();
    rand::manual_seed(1);
    let mut lin = Linear::new(128, 64, dev());
    let x = Tensor::randn(&[4, 128], dev());
    let want = no_grad(|| lin.forward(&x)).to_vec();
    lin.quantize_int4(32);
    let got = no_grad(|| lin.forward(&x)).to_vec();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rel = max_abs_diff(&got, &want) / scale;
    assert!(rel < 0.15, "int4 relative error {rel}");
    assert!(rel > 1e-4, "int4 should not be exact");
    assert_eq!(lin.trainable_parameters().len(), 1, "only the bias stays a parameter");
}

#[test]
fn int4_language_models() {
    let _l = lock();
    rand::manual_seed(2);
    let ids: Vec<usize> = (0..12).map(|i| (i * 5 + 1) % 41).collect();
    let t = Tensor::from_indices(&ids, &[1, 12], dev());
    let check = |reference: Vec<f32>, quant: Vec<f32>, what: &str| {
        let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_diff(&quant, &reference) / scale < 0.2, "{what}: error {}", max_abs_diff(&quant, &reference) / scale);
        // wherever the reference's top-1 leads by more than twice the error,
        // the quantized model must pick the same token
        let err = max_abs_diff(&quant, &reference);
        let mut decisive = 0;
        for (r, q) in reference.chunks(41).zip(quant.chunks(41)) {
            let best = argmax(r);
            let second = r.iter().enumerate().filter(|&(i, _)| i != best).fold(f32::MIN, |m, (_, &v)| m.max(v));
            if r[best] - second > 2.0 * err {
                decisive += 1;
                assert_eq!(argmax(q), best, "{what}: top-1 changed despite a clear margin");
            }
        }
        eprintln!("{what}: relative error {:.4}, {decisive}/12 positions with a clear margin", err / scale);
    };
    let mut l = Llama::new(LlamaConfig::new(41, 64, 2, 4, 2, 16), dev());
    let reference = no_grad(|| l.forward_tokens(&t)).to_vec();
    l.quantize_int4(32);
    check(reference, no_grad(|| l.forward_tokens(&t)).to_vec(), "llama");
    assert!(l.named_buffers().iter().any(|(n, _)| n == "model.layers.0.mlp.down_proj.weight_int4"));
    assert!(l.named_buffers().iter().any(|(n, _)| n == "lm_head.weight_int8"), "head stays int8");

    let mut g = GPT2::new(GPT2Config { vocab_size: 41, n_positions: 16, n_embd: 64, n_layer: 2, n_head: 2 }, dev());
    let reference = no_grad(|| g.forward_tokens(&t)).to_vec();
    g.quantize_int4(32);
    check(reference, no_grad(|| g.forward_tokens(&t)).to_vec(), "gpt2");
}

#[test]
fn lora_on_int4_base_trains() {
    let _l = lock();
    rand::manual_seed(3);
    let mut m = Llama::new(LlamaConfig::new(23, 32, 2, 4, 2, 16), dev());
    m.quantize_int4(16);
    assert_eq!(m.add_lora(4, 8.0, &["self_attn", "mlp"]), 14);
    let seq = [3usize, 1, 4, 1, 5, 9, 2, 6, 5];
    let inp = Tensor::from_indices(&seq[..8], &[1, 8], dev());
    let loss = || F::cross_entropy(&m.forward_tokens(&inp).reshape(&[8, 23]), &seq[1..], 0.0);
    let mut opt = Adam::new(m.trainable_parameters(), 0.02);
    let first = no_grad(|| loss().item());
    for _ in 0..60 {
        opt.zero_grad();
        loss().backward();
        opt.step();
    }
    let last = no_grad(|| loss().item());
    assert!(last < 0.5 * first, "loss {first} -> {last}");
}

#[test]
#[should_panic(expected = "must be even and divide")]
fn int4_group_must_divide_the_input_width() {
    Linear::new(30, 4, Device::Cpu).quantize_int4(16);
}
