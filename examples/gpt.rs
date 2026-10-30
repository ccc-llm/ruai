//! Character-level GPT (decoder-only Transformer): train, save, reload, generate.
//!
//!     cargo run --release --example gpt [path/to/text.txt]
//!     cargo run --release --features mps --example gpt -- --device mps
//!     cargo run --release --example gpt -- --accum 4     # gradient accumulation (4 micro-batches per step)
//!     cargo run --release --example gpt -- --resume      # continue from runs/gpt/last.safetensors

use rustnn::nn::{GPTConfig, Module, GPT};
use rustnn::checkpoint::{self, TrainState};
use rustnn::optim::lr::{CosineLr, LrScheduler};
use rustnn::optim::{clip_grad_norm, Adam, Optimizer};
use rustnn::prelude::*;
use rustnn::{rand, serialize};
use std::time::Instant;

const DEFAULT_TEXT: &str = "\
the quick brown fox jumps over the lazy dog. \
a neural network is a function made of layers. \
each layer multiplies by weights and adds a bias. \
training adjusts the weights to make the loss small. \
the gradient tells us which way the loss goes up, so we step the other way. \
attention lets every token look at the tokens before it. \
convolution slides a small filter over an image. \
rust is fast and safe, and now it can learn too. ";

/// `--device cpu|mps|auto` (or the `RUSTNN_DEVICE` env var); defaults to CPU.
fn device_arg() -> Device {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--device") {
        Some(i) => args.get(i + 1).expect("--device needs a value").parse().unwrap(),
        None => Device::from_env(),
    }
}

fn main() {
    rand::manual_seed(0);
    let device = device_arg();
    println!("device: {device}");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let accum: usize = args.iter().position(|a| a == "--accum").map_or(1, |i| args[i + 1].parse().expect("--accum N"));
    let resume = args.iter().any(|a| a == "--resume");
    // first positional argument (not a flag or a flag's value) is the text file
    let takes_value = |a: &str| a == "--device" || a == "--accum";
    let path = args.iter().enumerate().find(|(i, a)| !a.starts_with("--") && (*i == 0 || !takes_value(&args[i - 1])));
    let text = match path.map(|(_, p)| p.clone()) {
        Some(p) => std::fs::read_to_string(p).expect("read text file"),
        None => DEFAULT_TEXT.repeat(4),
    };

    // character vocabulary
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();
    let stoi = |c: char| vocab.binary_search(&c).unwrap();
    let data: Vec<usize> = text.chars().map(stoi).collect();
    println!("text: {} chars, vocab: {}", data.len(), vocab.len());

    let cfg = GPTConfig { vocab_size: vocab.len(), max_len: 64, dim: 64, n_heads: 4, n_layers: 2, dropout: 0.0 };
    let model = GPT::new(cfg, device);
    println!("model parameters: {}", model.num_parameters());

    let (batch, block, steps) = (16, cfg.max_len, 600);
    let base_lr = 3e-3;
    let mut opt = Adam::adamw(model.parameters(), base_lr, 0.01);
    let sched = CosineLr { base_lr, min_lr: base_lr * 0.1, warmup: 30, total: steps };
    let ckpt = "runs/gpt/last.safetensors";
    let mut state = TrainState::default();
    if resume && std::path::Path::new(ckpt).exists() {
        state = checkpoint::load(ckpt, &model, &mut opt).expect("load checkpoint");
        println!("resumed from {ckpt} at step {}", state.step);
    }
    if accum > 1 {
        println!("gradient accumulation: {accum} micro-batches of {batch} per step");
    }
    let t0 = Instant::now();
    while (state.step as usize) < steps {
        let step = state.step as usize;
        sched.apply(&mut opt, step);
        opt.zero_grad();
        let mut step_loss = 0.0;
        for _ in 0..accum {
            // random windows of the text
            let mut xs = Vec::with_capacity(batch * block);
            let mut ys = Vec::with_capacity(batch * block);
            for _ in 0..batch {
                let s = rand::randint(data.len() - block - 1);
                xs.extend_from_slice(&data[s..s + block]);
                ys.extend_from_slice(&data[s + 1..s + block + 1]);
            }
            let logits = model.forward_ids(&xs, batch, block).reshape(&[batch * block, cfg.vocab_size]);
            // gradients add up in .grad across backward() calls; scale so the
            // step uses the mean over all micro-batches
            let loss = F::cross_entropy(&logits, &ys, 0.0) * (1.0 / accum as f32);
            loss.backward();
            step_loss += loss.item();
        }
        clip_grad_norm(opt.params(), 1.0);
        opt.step();
        state.step += 1;
        if step.is_multiple_of(50) || step == steps - 1 {
            println!("step {step:4}  loss {step_loss:.4}  lr {:.5}  ({:.1}s)", opt.lr(), t0.elapsed().as_secs_f32());
        }
        if state.step.is_multiple_of(200) || state.step as usize == steps {
            checkpoint::save(ckpt, &model, &opt, &state).expect("save checkpoint");
        }
    }

    let path = "gpt.safetensors";
    serialize::save(path, &model).expect("save");

    // inference with a freshly loaded model
    let restored = GPT::new(cfg, device);
    serialize::load(path, &restored).expect("load");
    restored.eval();
    for prompt in ["the ", "attention "] {
        let ids: Vec<usize> = prompt.chars().map(stoi).collect();
        let out = restored.generate(&ids, 120, 0.0);
        let s: String = out.iter().map(|&i| vocab[i]).collect();
        println!("\n[greedy] {s}");
    }
    let ids: Vec<usize> = "rust ".chars().map(stoi).collect();
    let s: String = restored.generate(&ids, 120, 0.8).iter().map(|&i| vocab[i]).collect();
    println!("\n[T=0.8] {s}");
}
