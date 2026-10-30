//! Micro / macro benchmarks.
//!
//!     cargo run --release --example bench [cpu|mps]
//!     cargo run --release --features mps --example bench mps

use ruai::nn::{self, GPTConfig, Module, Sequential, GPT};
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use ruai::rand;
use std::time::Instant;

/// Mean wall time (ms) per call of `f` over `iters` back-to-back runs, after a
/// warmup. The device is synchronized once at the end (as with
/// `torch.cuda.synchronize()`), so GPU numbers measure throughput, not latency.
fn time_ms(dev: Device, iters: usize, mut f: impl FnMut()) -> f64 {
    // warm up (lets GPU clocks ramp up and fills buffer pools)
    for _ in 0..iters.div_ceil(2) {
        f();
    }
    dev.synchronize();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    dev.synchronize();
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    rand::manual_seed(0);
    let dev: Device = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or_else(Device::from_env);
    println!("device: {dev}");

    println!("== matmul ==");
    for &n in &[128usize, 256, 512, 1024] {
        let a = Tensor::randn(&[n, n], dev);
        let b = Tensor::randn(&[n, n], dev);
        let ms = time_ms(dev, 20, || {
            a.matmul(&b);
        });
        println!("  {n:4}x{n:<4}  {ms:8.3} ms  {:7.1} GFLOP/s", 2.0 * (n as f64).powi(3) / ms / 1e6);
    }
    let a = Tensor::randn(&[32, 8, 64, 64], dev);
    let b = Tensor::randn(&[32, 8, 64, 64], dev);
    let ms = time_ms(dev, 20, || {
        a.matmul(&b.t());
    });
    println!("  batched [32,8,64,64]@T  {ms:8.3} ms");

    println!("== elementwise / reductions (1M elements) ==");
    let x = Tensor::randn(&[1024, 1024], dev);
    let bias = Tensor::randn(&[1024], dev);
    println!("  add bcast   {:8.3} ms", time_ms(dev, 20, || drop(&x + &bias)));
    println!("  gelu        {:8.3} ms", time_ms(dev, 20, || drop(x.gelu())));
    println!("  sum(-1)     {:8.3} ms", time_ms(dev, 20, || drop(x.sum(-1, false))));
    println!("  softmax     {:8.3} ms", time_ms(dev, 20, || drop(F::softmax(&x, -1))));
    println!("  layer_norm  {:8.3} ms", time_ms(dev, 20, || drop(F::layer_norm(&x, None, None, 1e-5))));
    println!("  transpose   {:8.3} ms", time_ms(dev, 20, || drop(x.t())));

    println!("== conv2d fwd+bwd [32,16,32,32] * [32,16,3,3] ==");
    let conv = nn::Conv2d::new(16, 32, 3, 1, 1, dev);
    let xi = Tensor::randn(&[32, 16, 32, 32], dev).requires_grad_();
    println!(
        "  {:8.3} ms",
        time_ms(dev, 20, || {
            conv.zero_grad();
            conv.forward(&xi).sum_all().backward();
        })
    );

    println!("== CNN train step (batch 64, 28x28) ==");
    let cnn = Sequential::new()
        .add(nn::Conv2d::new(1, 8, 3, 1, 1, dev))
        .add(nn::BatchNorm2d::new(8, dev))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Conv2d::new(8, 16, 3, 1, 1, dev))
        .add(nn::BatchNorm2d::new(16, dev))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Flatten)
        .add(nn::Linear::new(16 * 7 * 7, 64, dev))
        .add(nn::ReLU)
        .add(nn::Linear::new(64, 10, dev));
    let mut opt = Adam::new(cnn.parameters(), 1e-3);
    let xb = Tensor::randn(&[64, 1, 28, 28], dev);
    let yb: Vec<usize> = (0..64).map(|i| i % 10).collect();
    println!(
        "  {:8.3} ms/step",
        time_ms(dev, 20, || {
            opt.zero_grad();
            F::cross_entropy(&cnn.forward(&xb), &yb, 0.0).backward();
            opt.step();
        })
    );

    println!("== GPT train step (B=16, T=64, d=64, L=2, V=64) ==");
    let cfg = GPTConfig { vocab_size: 64, max_len: 64, dim: 64, n_heads: 4, n_layers: 2, dropout: 0.0 };
    let gpt = GPT::new(cfg, dev);
    let mut opt = Adam::new(gpt.parameters(), 1e-3);
    let ids: Vec<usize> = (0..16 * 64).map(|i| (i * 7) % 64).collect();
    println!(
        "  {:8.3} ms/step",
        time_ms(dev, 20, || {
            opt.zero_grad();
            let logits = gpt.forward_ids(&ids, 16, 64).reshape(&[16 * 64, 64]);
            F::cross_entropy(&logits, &ids, 0.0).backward();
            opt.step();
        })
    );

    let n = 2048usize;
    for d in [DType::F32, DType::F16, DType::BF16] {
        let a = Tensor::randn(&[n, n], dev).to_dtype(d);
        let b = Tensor::randn(&[n, n], dev).to_dtype(d);
        let ms = time_ms(dev, 10, || drop(a.matmul(&b)));
        println!("== matmul {n}x{n} {d}: {ms:8.3} ms  {:7.1} GFLOP/s", 2.0 * (n as f64).powi(3) / ms / 1e6);
    }

    let cfg = GPTConfig { vocab_size: 256, max_len: 128, dim: 256, n_heads: 8, n_layers: 4, dropout: 0.0 };
    let ids: Vec<usize> = (0..8 * 128).map(|i| (i * 7) % 256).collect();
    for amp in [None, Some(DType::F16), Some(DType::BF16)] {
        rand::manual_seed(0);
        let gpt = GPT::new(cfg, dev);
        let mut opt = Adam::new(gpt.parameters(), 1e-3);
        let mut scaler = ruai::amp::GradScaler::new();
        let label = amp.map_or("f32".to_string(), |d| format!("autocast {d}"));
        println!("== GPT train step (B=8, T=128, d=256, L=4, V=256), {label} ==");
        let ms = time_ms(dev, 10, || {
            opt.zero_grad();
            let step = || F::cross_entropy(&gpt.forward_ids(&ids, 8, 128).reshape(&[8 * 128, 256]), &ids, 0.0);
            match amp {
                // f16 needs loss scaling; bf16 has the f32 exponent range
                Some(DType::F16) => {
                    scaler.scale(&ruai::autocast(DType::F16, step)).backward();
                    scaler.step(&mut opt);
                }
                Some(d) => {
                    ruai::autocast(d, step).backward();
                    opt.step();
                }
                None => {
                    step().backward();
                    opt.step();
                }
            }
        });
        println!("  {ms:8.3} ms/step");
    }

    // decoding-style matvec through a 2048 x 2048 weight: memory bound, so
    // smaller weights read faster
    println!("== weight-only quantized matvec [1, 2048] @ [2048, 2048] ==");
    for q in ["f32", "int8", "int4"] {
        rand::manual_seed(0);
        let mut lin = ruai::nn::Linear::with_bias(2048, 2048, false, dev);
        match q {
            "int8" => lin.quantize_int8(),
            "int4" => lin.quantize_int4(32),
            _ => {}
        }
        let x = Tensor::randn(&[1, 2048], dev);
        let ms = no_grad(|| time_ms(dev, 50, || drop(lin.forward(&x))));
        println!("  {q:5} {ms:8.3} ms");
    }
}
