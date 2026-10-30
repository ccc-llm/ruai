//! Character-level Llama (RMSNorm + RoPE + SwiGLU + grouped-query attention):
//! train, export in Hugging Face format, reload, generate with a KV cache.
//!
//!     cargo run --release --example llama [path/to/text.txt]
//!     cargo run --release --features mps --example llama -- --device mps

use ruai::generation::Sampler;
use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::nn::Module;
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::optim::{clip_grad_norm, Adam, Optimizer};
use ruai::prelude::*;
use ruai::rand;
use std::time::Instant;

const DEFAULT_TEXT: &str = "\
the quick brown fox jumps over the lazy dog. \
a neural network is a function made of layers. \
rotary embeddings turn each position into an angle. \
rms norm scales a vector by its root mean square. \
grouped query attention lets many heads share one key and value. \
the gated feed forward multiplies a silu gate by a linear branch. \
rust is fast and safe, and now it can learn too. ";

fn main() {
    rand::manual_seed(0);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = match args.iter().position(|a| a == "--device") {
        Some(i) => args[i + 1].parse().unwrap(),
        None => Device::from_env(),
    };
    println!("device: {device}");
    let path = args.iter().enumerate().find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--device"));
    let text = match path {
        Some((_, p)) => std::fs::read_to_string(p).expect("read text file"),
        None => DEFAULT_TEXT.repeat(4),
    };

    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();
    let stoi = |c: char| vocab.binary_search(&c).unwrap();
    let data: Vec<usize> = text.chars().map(stoi).collect();
    println!("text: {} chars, vocab: {}", data.len(), vocab.len());

    // 4 query heads share 2 key/value heads
    let cfg = LlamaConfig::new(vocab.len(), 64, 2, 4, 2, 64);
    let model = Llama::new(cfg, device);
    println!("model parameters: {} (SwiGLU hidden {})", model.num_parameters(), cfg.intermediate_size);

    let (batch, block, steps) = (16, cfg.max_position_embeddings, 600);
    let base_lr = 3e-3;
    let mut opt = Adam::adamw(model.parameters(), base_lr, 0.01);
    let sched = CosineLr { base_lr, min_lr: base_lr * 0.1, warmup: 30, total: steps };
    let t0 = Instant::now();
    for step in 0..steps {
        sched.apply(&mut opt, step);
        let mut xs = Vec::with_capacity(batch * block);
        let mut ys = Vec::with_capacity(batch * block);
        for _ in 0..batch {
            let s = rand::randint(data.len() - block - 1);
            xs.extend_from_slice(&data[s..s + block]);
            ys.extend_from_slice(&data[s + 1..s + block + 1]);
        }
        opt.zero_grad();
        let logits = model.forward_tokens(&Tensor::from_indices(&xs, &[batch, block], device));
        let loss = F::cross_entropy(&logits.reshape(&[batch * block, cfg.vocab_size]), &ys, 0.0);
        loss.backward();
        clip_grad_norm(opt.params(), 1.0);
        opt.step();
        if step.is_multiple_of(50) || step == steps - 1 {
            println!("step {step:4}  loss {:.4}  lr {:.5}  ({:.1}s)", loss.item(), opt.lr(), t0.elapsed().as_secs_f32());
        }
    }

    // export as a Hugging Face checkpoint (config.json + model.safetensors) and reload it
    let dir = "runs/llama";
    model.save_pretrained(dir).expect("save");
    let restored = Llama::from_pretrained(dir, device).expect("load");
    println!("\nsaved and reloaded {dir}/ (Hugging Face LlamaForCausalLM layout)");
    for prompt in ["the ", "rotary "] {
        let ids: Vec<usize> = prompt.chars().map(stoi).collect();
        let out = restored.generate(&ids, 120, &mut Sampler::greedy());
        println!("\n[greedy] {}", out.iter().map(|&i| vocab[i]).collect::<String>());
    }
    let ids: Vec<usize> = "rust ".chars().map(stoi).collect();
    let out = restored.generate(&ids, 120, &mut Sampler::new(0.8, 0, 0.95, 1));
    println!("\n[T=0.8, top-p 0.95] {}", out.iter().map(|&i| vocab[i]).collect::<String>());
}
