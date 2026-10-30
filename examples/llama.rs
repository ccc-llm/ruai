//! Character-level Llama (RMSNorm + RoPE + SwiGLU + grouped-query attention):
//! train, export in Hugging Face format, reload, generate with a KV cache.
//!
//!     cargo run --release --example llama [path/to/text.txt]
//!     cargo run --release --features mps --example llama -- --device mps
//!
//! With `--model DIR` it instead loads a pretrained Hugging Face Llama-family
//! checkpoint (`config.json`, `model.safetensors`, `tokenizer.json`) and
//! continues `--prompt` (see `scripts/download_smollm.sh`):
//!
//!     ./scripts/download_smollm.sh
//!     cargo run --release --example llama -- --model data/smollm2-135m --prompt "The capital of France is"

use ruai::generation::Sampler;
use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::nn::Module;
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::optim::{clip_grad_norm, Adam, Optimizer};
use ruai::prelude::*;
use ruai::rand;
use ruai::tokenizer::HfTokenizer;
use std::io::Write;
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
    let value = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    if let Some(dir) = value("--model") {
        let prompt = value("--prompt").unwrap_or_else(|| "Once upon a time".into());
        let n = value("-n").map_or(60, |n| n.parse().expect("-n N"));
        return pretrained(&dir, &prompt, n, device);
    }
    let takes_value = |a: &str| matches!(a, "--device" | "--model" | "--prompt" | "-n");
    let path = args.iter().enumerate().find(|(i, a)| !a.starts_with("--") && (*i == 0 || !takes_value(&args[i - 1])));
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

/// Load a Hugging Face checkpoint and stream a greedy continuation of `prompt`.
fn pretrained(dir: &str, prompt: &str, n: usize, device: Device) {
    let t0 = Instant::now();
    let tok = HfTokenizer::from_dir(dir).expect("tokenizer.json");
    let model = Llama::from_pretrained(dir, device).expect("load model");
    let c = &model.config;
    println!(
        "loaded {dir}: {} layers, hidden {}, {} heads / {} kv heads, vocab {} ({:.1}M parameters, {:.1}s)",
        c.num_hidden_layers,
        c.hidden_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.vocab_size,
        model.num_parameters() as f32 / 1e6,
        t0.elapsed().as_secs_f32()
    );
    let ids = tok.encode(prompt, true);
    println!("prompt tokens: {ids:?}\n");
    print!("{prompt}");
    let mut shown = tok.decode(&ids, true);
    let mut all = ids.clone();
    let t1 = Instant::now();
    let out = model.generate_with(&ids, n, &mut Sampler::greedy(), |next| {
        // re-decode the whole text and print the new suffix (handles merged UTF-8 bytes)
        all.push(next);
        let text = tok.decode(&all, true);
        if let Some(new) = text.strip_prefix(&shown) {
            print!("{new}");
            std::io::stdout().flush().ok();
        }
        shown = text;
        true
    });
    let new = out.len() - ids.len();
    println!("\n\n{new} tokens in {:.2}s ({:.1} tokens/s)", t1.elapsed().as_secs_f32(), new as f32 / t1.elapsed().as_secs_f32());
}
