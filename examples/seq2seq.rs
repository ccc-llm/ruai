//! Encoder-decoder Transformer learning to sort digit strings
//! ("4071" -> "0147"): padded batches with source padding masks, an
//! ignore-index loss and cached greedy decoding.
//!
//!     cargo run --release --example seq2seq
//!     cargo run --release --features mps --example seq2seq -- --device mps

use ruai::nn::seq2seq::{Seq2Seq, Seq2SeqConfig};
use ruai::nn::Module;
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use ruai::rand;
use std::time::Instant;

const PAD: usize = 0;
const BOS: usize = 1;
const EOS: usize = 2;
const MAX_DIGITS: usize = 8;

fn tok(d: u32) -> usize {
    d as usize + 3
}

fn digits(ids: &[usize]) -> String {
    ids.iter().filter(|&&i| i >= 3).map(|&i| char::from_digit((i - 3) as u32, 10).unwrap()).collect()
}

fn random_digits() -> Vec<usize> {
    let len = 3 + rand::randint(MAX_DIGITS - 2);
    (0..len).map(|_| tok(rand::randint(10) as u32)).collect()
}

/// Right-padded batch: (source, decoder input `BOS + y`, decoder target `y + EOS`).
fn batch(n: usize, device: Device) -> (Tensor, Tensor, Vec<usize>) {
    let (ls, lt) = (MAX_DIGITS, MAX_DIGITS + 1);
    let (mut src, mut tin, mut tout) = (vec![PAD; n * ls], vec![PAD; n * lt], vec![PAD; n * lt]);
    for i in 0..n {
        let s = random_digits();
        let mut y = s.clone();
        y.sort();
        src[i * ls..i * ls + s.len()].copy_from_slice(&s);
        tin[i * lt] = BOS;
        tin[i * lt + 1..i * lt + 1 + y.len()].copy_from_slice(&y);
        y.push(EOS);
        tout[i * lt..i * lt + y.len()].copy_from_slice(&y);
    }
    (Tensor::from_indices(&src, &[n, ls], device), Tensor::from_indices(&tin, &[n, lt], device), tout)
}

fn main() {
    rand::manual_seed(0);
    let args: Vec<String> = std::env::args().collect();
    let device = match args.iter().position(|a| a == "--device") {
        Some(i) => args[i + 1].parse().unwrap(),
        None => Device::from_env(),
    };
    println!("device: {device}");
    let cfg = Seq2SeqConfig {
        src_vocab: 13,
        tgt_vocab: 13,
        dim: 64,
        heads: 4,
        encoder_layers: 2,
        decoder_layers: 2,
        ff_hidden: 128,
        max_len: MAX_DIGITS + 1,
        dropout: 0.0,
    };
    let model = Seq2Seq::new(cfg, device);
    println!("model parameters: {}", model.num_parameters());

    let steps = 800;
    let mut opt = Adam::adamw(model.parameters(), 2e-3, 0.01);
    let sched = CosineLr { base_lr: 2e-3, min_lr: 1e-4, warmup: 50, total: steps };
    let t0 = Instant::now();
    for step in 0..steps {
        sched.apply(&mut opt, step);
        let (src, tin, tout) = batch(64, device);
        let mask = F::padding_mask(&src, PAD);
        opt.zero_grad();
        let logits = model.forward_pair(&src, &tin, Some(&mask));
        let (b, t, v) = (logits.dim(0), logits.dim(1), logits.dim(2));
        let loss = F::cross_entropy_ignore(&logits.reshape(&[b * t, v]), &tout, PAD);
        loss.backward();
        opt.step();
        if step.is_multiple_of(100) || step == steps - 1 {
            println!("step {step:4}  loss {:.4}  ({:.1}s)", loss.item(), t0.elapsed().as_secs_f32());
        }
    }

    model.eval();
    let mut correct = 0;
    let n = 200;
    for i in 0..n {
        let s = random_digits();
        let mut want = s.clone();
        want.sort();
        want.push(EOS);
        let got = model.greedy(&s, BOS, Some(EOS), MAX_DIGITS + 1);
        correct += (got == want) as usize;
        if i < 6 {
            println!("{:>8} -> {}", digits(&s), digits(&got));
        }
    }
    println!("exact-match accuracy: {:.1}% ({correct}/{n})", 100.0 * correct as f32 / n as f32);
}
