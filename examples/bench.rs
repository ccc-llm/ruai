//! Micro / macro benchmarks.
//!
//!     cargo run --release --example bench

use rustnn::nn::{self, GPTConfig, Module, Sequential, GPT};
use rustnn::optim::{Adam, Optimizer};
use rustnn::prelude::*;
use rustnn::rand;
use std::time::Instant;

/// Median wall time (ms) of `f` over `iters` runs after one warmup.
fn time_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut ts: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ts.sort_by(|a, b| a.total_cmp(b));
    ts[ts.len() / 2]
}

fn main() {
    rand::manual_seed(0);
    let dev = Device::Cpu;

    println!("== matmul ==");
    for &n in &[128usize, 256, 512, 1024] {
        let a = Tensor::randn(&[n, n], dev);
        let b = Tensor::randn(&[n, n], dev);
        let ms = time_ms(5, || {
            a.matmul(&b);
        });
        println!("  {n:4}x{n:<4}  {ms:8.3} ms  {:7.1} GFLOP/s", 2.0 * (n as f64).powi(3) / ms / 1e6);
    }
    let a = Tensor::randn(&[32, 8, 64, 64], dev);
    let b = Tensor::randn(&[32, 8, 64, 64], dev);
    let ms = time_ms(5, || {
        a.matmul(&b.t());
    });
    println!("  batched [32,8,64,64]@T  {ms:8.3} ms");

    println!("== elementwise / reductions (1M elements) ==");
    let x = Tensor::randn(&[1024, 1024], dev);
    let bias = Tensor::randn(&[1024], dev);
    println!("  add bcast   {:8.3} ms", time_ms(10, || drop(&x + &bias)));
    println!("  gelu        {:8.3} ms", time_ms(10, || drop(x.gelu())));
    println!("  sum(-1)     {:8.3} ms", time_ms(10, || drop(x.sum(-1, false))));
    println!("  softmax     {:8.3} ms", time_ms(10, || drop(F::softmax(&x, -1))));
    println!("  layer_norm  {:8.3} ms", time_ms(10, || drop(F::layer_norm(&x, None, None, 1e-5))));
    println!("  transpose   {:8.3} ms", time_ms(10, || drop(x.t())));

    println!("== conv2d fwd+bwd [32,16,32,32] * [32,16,3,3] ==");
    let conv = nn::Conv2d::new(16, 32, 3, 1, 1, dev);
    let xi = Tensor::randn(&[32, 16, 32, 32], dev).requires_grad_();
    println!(
        "  {:8.3} ms",
        time_ms(5, || {
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
        time_ms(10, || {
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
        time_ms(10, || {
            opt.zero_grad();
            let logits = gpt.forward_ids(&ids, 16, 64).reshape(&[16 * 64, 64]);
            F::cross_entropy(&logits, &ids, 0.0).backward();
            opt.step();
        })
    );

    println!("== GPT train step (B=8, T=128, d=256, L=4, V=256) ==");
    let cfg = GPTConfig { vocab_size: 256, max_len: 128, dim: 256, n_heads: 8, n_layers: 4, dropout: 0.0 };
    let gpt = GPT::new(cfg, dev);
    let mut opt = Adam::new(gpt.parameters(), 1e-3);
    let ids: Vec<usize> = (0..8 * 128).map(|i| (i * 7) % 256).collect();
    println!(
        "  {:8.3} ms/step",
        time_ms(5, || {
            opt.zero_grad();
            let logits = gpt.forward_ids(&ids, 8, 128).reshape(&[8 * 128, 256]);
            F::cross_entropy(&logits, &ids, 0.0).backward();
            opt.step();
        })
    );
}
