//! Text generation with pretrained GPT-2 (Hugging Face weights).
//!
//!     ./scripts/download_gpt2.sh                      # once: fetch weights + tokenizer into data/gpt2
//!     cargo run --release --example gpt2 -- --prompt "The meaning of life is"
//!     cargo run --release --example gpt2 -- --prompt "Once upon a time" --temperature 0.8 --top-k 40
//!     cargo run --release --example gpt2 -- --int8                     # int8 weights
//!     cargo run --release --features mps --example gpt2 -- --device mps
//!     cargo run --release --example gpt2 -- --no-cache                 # recompute the context each step
//!
//! Options: --model DIR (default data/gpt2), --prompt TEXT, --max-tokens N (100),
//! --temperature T (0 = greedy, default 0.8), --top-k K (40), --top-p P (0.95),
//! --seed S, --device cpu|mps|cuda|auto, --int8, --no-cache.

use ruai::generation::Sampler;
use ruai::nn::gpt2::GPT2;
use ruai::tokenizer::{Gpt2Tokenizer, EOT};
use ruai::{no_grad, Device, Tensor};
use std::io::Write;
use std::time::Instant;

struct Args(Vec<String>);

impl Args {
    fn value(&self, flag: &str) -> Option<&str> {
        self.0.iter().position(|a| a == flag).map(|i| self.0.get(i + 1).unwrap_or_else(|| panic!("{flag} needs a value")).as_str())
    }
    fn get<T: std::str::FromStr>(&self, flag: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.value(flag).map_or(default, |v| v.parse().unwrap_or_else(|e| panic!("bad {flag}: {e:?}")))
    }
    fn has(&self, flag: &str) -> bool {
        self.0.iter().any(|a| a == flag)
    }
}

fn main() {
    let args = Args(std::env::args().skip(1).collect());
    let dir = args.value("--model").unwrap_or("data/gpt2").to_string();
    let prompt = args.value("--prompt").unwrap_or("The meaning of life is").to_string();
    let max_tokens: usize = args.get("--max-tokens", 100);
    let device: Device = args.value("--device").map_or_else(Device::from_env, |d| d.parse().unwrap());
    let seed: u64 = args.get("--seed", 0);
    let mut sampler = Sampler::new(args.get("--temperature", 0.8), args.get("--top-k", 40), args.get("--top-p", 0.95), seed);

    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        eprintln!("GPT-2 weights not found in {dir}. Download them with:\n\n    ./scripts/download_gpt2.sh\n");
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let tok = Gpt2Tokenizer::from_dir(&dir).expect("load tokenizer");
    let mut model = GPT2::from_pretrained(&dir, device).expect("load model");
    if args.has("--int8") {
        model.quantize_int8();
    }
    let c = model.config;
    eprintln!(
        "loaded GPT-2 ({} layers, d={}, {:.0}M params) on {device}{} in {:.1}s",
        c.n_layer,
        c.n_embd,
        ruai::nn::Module::num_parameters(&model) as f64 / 1e6,
        if args.has("--int8") { ", int8 weights" } else { "" },
        t0.elapsed().as_secs_f32()
    );

    let ids = tok.encode(&prompt);
    print!("{prompt}");
    std::io::stdout().flush().ok();
    // stream decoded text; hold back bytes of an incomplete UTF-8 character
    let mut pending: Vec<u8> = Vec::new();
    let mut emit = |id: usize| {
        pending.extend(tok.decode_bytes(&[id]));
        match std::str::from_utf8(&pending) {
            Ok(s) => {
                print!("{s}");
                pending.clear();
            }
            Err(e) if e.error_len().is_some() => {
                print!("{}", String::from_utf8_lossy(&pending));
                pending.clear();
            }
            Err(_) => {} // wait for the rest of the character
        }
        std::io::stdout().flush().ok();
        id != EOT
    };

    let t1 = Instant::now();
    let generated = if args.has("--no-cache") {
        // reference path: re-run the whole context for every token (O(n²))
        let mut all = ids.clone();
        no_grad(|| {
            for _ in 0..max_tokens {
                let ctx = &all[all.len().saturating_sub(c.n_positions)..];
                let logits = model.forward_tokens(&Tensor::from_indices(ctx, &[1, ctx.len()], device));
                let v = logits.dim(-1);
                let next = sampler.sample(&logits.narrow(1, ctx.len() - 1, 1).reshape(&[v]).to_vec());
                all.push(next);
                if !emit(next) {
                    break;
                }
            }
        });
        all.len() - ids.len()
    } else {
        model.generate_with(&ids, max_tokens, &mut sampler, &mut emit).len() - ids.len()
    };
    let secs = t1.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "\n{generated} tokens in {secs:.2}s: {:.1} tokens/s ({} prompt tokens, {})",
        generated as f64 / secs,
        ids.len(),
        if args.has("--no-cache") { "no KV cache" } else { "KV cache" }
    );
}
