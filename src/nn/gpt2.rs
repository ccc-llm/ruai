//! OpenAI GPT-2, parameter-compatible with Hugging Face `GPT2LMHeadModel`
//! (`openai-community/gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
//!
//! Parameter names match the Hugging Face checkpoints (`wte.weight`,
//! `h.0.attn.c_attn.weight`, ...) and HF's `Conv1D` stores weights as
//! `[in, out]` — the same layout as [`Linear`] — so `model.safetensors`
//! loads without any conversion. The output head is tied to `wte`.
//!
//! ```no_run
//! use ruai::nn::gpt2::GPT2;
//! use ruai::tokenizer::Gpt2Tokenizer;
//! use ruai::Device;
//! let model = GPT2::from_pretrained("data/gpt2", Device::Cpu).unwrap();
//! let tok = Gpt2Tokenizer::from_dir("data/gpt2").unwrap();
//! let ids = model.generate(&tok.encode("The dog"), 20, &mut ruai::generation::Sampler::greedy());
//! println!("{}", tok.decode(&ids));
//! ```

use super::layers::{Embedding, LayerNorm, Linear};
use super::module::{prefixed, Module};
use super::transformer::{KvCache, LayerKv};
use crate::functional as F;
use crate::generation::Sampler;
use crate::serialize;
use crate::tensor::Tensor;
use crate::Device;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// GPT-2 hyperparameters (names as in Hugging Face `GPT2Config`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GPT2Config {
    /// Vocabulary size (50257 for the OpenAI tokenizer).
    pub vocab_size: usize,
    /// Maximum context length.
    pub n_positions: usize,
    /// Model width.
    pub n_embd: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Attention heads per block.
    pub n_head: usize,
}

impl GPT2Config {
    /// GPT-2 small (124M).
    pub fn gpt2() -> Self {
        GPT2Config { vocab_size: 50257, n_positions: 1024, n_embd: 768, n_layer: 12, n_head: 12 }
    }
    /// GPT-2 medium (355M).
    pub fn medium() -> Self {
        GPT2Config { n_embd: 1024, n_layer: 24, n_head: 16, ..Self::gpt2() }
    }
    /// GPT-2 large (774M).
    pub fn large() -> Self {
        GPT2Config { n_embd: 1280, n_layer: 36, n_head: 20, ..Self::gpt2() }
    }
    /// GPT-2 XL (1.5B).
    pub fn xl() -> Self {
        GPT2Config { n_embd: 1600, n_layer: 48, n_head: 25, ..Self::gpt2() }
    }
}

#[derive(Clone)]
struct Attention {
    c_attn: Linear, // [n_embd, 3 * n_embd]: q, k, v
    c_proj: Linear,
    n_head: usize,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

#[derive(Clone)]
struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

/// GPT-2 language model (inference and fine-tuning).
pub struct GPT2 {
    /// Architecture hyperparameters.
    pub config: GPT2Config,
    /// Token embeddings `[vocab, n_embd]` (also the tied output head).
    pub wte: Embedding,
    /// Learned position embeddings `[n_positions, n_embd]`.
    pub wpe: Embedding,
    blocks: Vec<Block>,
    /// Final layer norm.
    pub ln_f: LayerNorm,
    /// int8 copy of the tied head (`wteᵀ`), set by [`GPT2::quantize_int8`].
    head_int8: Option<Linear>,
    gradient_checkpointing: AtomicBool,
}

impl Attention {
    fn forward(&self, x: &Tensor, kv: Option<&mut LayerKv>) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.n_head;
        let qkv = self.c_attn.forward(x);
        let heads = |i: usize| qkv.narrow(-1, i * d, d).reshape(&[b, t, h, d / h]).permute(&[0, 2, 1, 3]);
        let (q, k, v) = (heads(0), heads(1), heads(2));
        let (k, v, past) = match kv {
            Some(kv) => {
                let past = kv.len();
                let (k, v) = kv.append(k, v);
                (k, v, past)
            }
            None => (k, v, 0),
        };
        let mask = (t > 1).then(|| F::causal_mask_offset(t, past, x.device()));
        let o = F::scaled_dot_product_attention(&q, &k, &v, mask.as_ref(), 0.0, false);
        self.c_proj.forward(&o.permute(&[0, 2, 1, 3]).reshape(&[b, t, d]))
    }
}

impl Block {
    fn linears_mut(&mut self) -> [(&'static str, &mut Linear); 4] {
        [
            ("attn.c_attn", &mut self.attn.c_attn),
            ("attn.c_proj", &mut self.attn.c_proj),
            ("mlp.c_fc", &mut self.mlp.c_fc),
            ("mlp.c_proj", &mut self.mlp.c_proj),
        ]
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.ln_1.parameters();
        for l in [&self.attn.c_attn, &self.attn.c_proj, &self.mlp.c_fc, &self.mlp.c_proj] {
            v.extend(l.parameters());
        }
        v.extend(self.ln_2.parameters());
        v
    }

    fn forward(&self, x: &Tensor, kv: Option<&mut LayerKv>) -> Tensor {
        let x = x + self.attn.forward(&self.ln_1.forward(x), kv);
        let h = self.mlp.c_fc.forward(&self.ln_2.forward(&x)).gelu(); // "gelu_new" (tanh)
        &x + self.mlp.c_proj.forward(&h)
    }
}

impl GPT2 {
    /// Randomly initialized model (std 0.02, as in GPT-2).
    pub fn new(config: GPT2Config, device: Device) -> Self {
        let d = config.n_embd;
        assert_eq!(d % config.n_head, 0, "n_embd must be divisible by n_head");
        let blocks = (0..config.n_layer)
            .map(|_| Block {
                ln_1: LayerNorm::new(d, device),
                attn: Attention {
                    c_attn: Linear::new(d, 3 * d, device),
                    c_proj: Linear::new(d, d, device),
                    n_head: config.n_head,
                },
                ln_2: LayerNorm::new(d, device),
                mlp: Mlp { c_fc: Linear::new(d, 4 * d, device), c_proj: Linear::new(4 * d, d, device) },
            })
            .collect();
        GPT2 {
            config,
            wte: Embedding::new(config.vocab_size, d, device),
            wpe: Embedding::new(config.n_positions, d, device),
            blocks,
            ln_f: LayerNorm::new(d, device),
            head_int8: None,
            gradient_checkpointing: AtomicBool::new(false),
        }
    }

    /// Load a Hugging Face checkpoint: `dir/model.safetensors` (or a path to
    /// the file itself). The architecture is inferred from the tensor shapes
    /// (heads = n_embd / 64, as in every GPT-2 size).
    pub fn from_pretrained(path: impl AsRef<Path>, device: Device) -> io::Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() { path.join("model.safetensors") } else { path.to_path_buf() };
        let (tensors, _) = serialize::load_safetensors(&file)?;
        // some exports prefix everything with "transformer."; the causal-mask
        // buffers (`attn.bias`, `attn.masked_bias`) are not parameters
        let tensors: Vec<(String, Tensor)> = tensors
            .into_iter()
            .map(|(n, t)| (n.strip_prefix("transformer.").unwrap_or(&n).to_string(), t))
            .filter(|(n, _)| !n.ends_with(".attn.bias") && !n.ends_with(".attn.masked_bias") && n != "lm_head.weight")
            .collect();
        let shape = |name: &str| {
            tensors
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.shape().to_vec())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("checkpoint has no '{name}'")))
        };
        let wte = shape("wte.weight")?;
        let wpe = shape("wpe.weight")?;
        let n_layer = tensors.iter().filter(|(n, _)| n.ends_with(".ln_1.weight")).count();
        let config =
            GPT2Config { vocab_size: wte[0], n_positions: wpe[0], n_embd: wte[1], n_layer, n_head: wte[1] / 64 };
        let model = GPT2::new(config, device);
        serialize::load_state_dict(&model, &tensors, true)?;
        Ok(model)
    }

    /// Logits `[B, T, vocab]` for `tokens` `[B, T]` (no cache).
    pub fn forward_tokens(&self, tokens: &Tensor) -> Tensor {
        self.run(tokens, None)
    }

    /// Logits for new tokens after the positions already in `cache` (which is extended).
    pub fn forward_cached(&self, tokens: &Tensor, cache: &mut KvCache) -> Tensor {
        if cache.layers.len() != self.blocks.len() {
            *cache = KvCache::new(self.blocks.len());
        }
        self.run(tokens, Some(cache))
    }

    fn run(&self, tokens: &Tensor, mut cache: Option<&mut KvCache>) -> Tensor {
        let t = tokens.shape()[1];
        let start = cache.as_ref().map_or(0, |c| c.len());
        assert!(start + t <= self.config.n_positions, "context {} exceeds n_positions {}", start + t, self.config.n_positions);
        let pos = Tensor::from_indices(&(start..start + t).collect::<Vec<_>>(), &[t], tokens.device());
        let mut x = self.wte.forward(tokens) + self.wpe.forward(&pos);
        let checkpoint = cache.is_none() && self.gradient_checkpointing.load(Ordering::Relaxed) && crate::is_grad_enabled();
        for (i, blk) in self.blocks.iter().enumerate() {
            x = if checkpoint {
                let b = blk.clone();
                F::grad_checkpoint(&[&x], blk.parameters(), move |xs| b.forward(&xs[0], None))
            } else {
                blk.forward(&x, cache.as_mut().map(|c| &mut c.layers[i]))
            };
        }
        let x = self.ln_f.forward(&x);
        match &self.head_int8 {
            Some(head) => head.forward(&x),
            None => x.matmul_nt(&self.wte.weight), // tied head: x @ wteᵀ
        }
    }

    /// Generate `n` tokens after `prompt` with a KV cache. Stops early at
    /// `<|endoftext|>` (50256) only if it is sampled; the full sequence
    /// (prompt included) is returned.
    pub fn generate(&self, prompt: &[usize], n: usize, sampler: &mut Sampler) -> Vec<usize> {
        self.generate_with(prompt, n, sampler, |_| true)
    }

    /// Like [`GPT2::generate`], calling `on_token` after each new token;
    /// returning `false` stops generation.
    pub fn generate_with(
        &self,
        prompt: &[usize],
        n: usize,
        sampler: &mut Sampler,
        mut on_token: impl FnMut(usize) -> bool,
    ) -> Vec<usize> {
        let dev = self.wte.weight.device();
        let max = self.config.n_positions;
        crate::generation::generate_cached(prompt, n, max, dev, sampler, |t, c| self.forward_cached(t, c), &mut on_token)
    }

    /// Recompute each block's activations during the backward pass instead
    /// of storing them ([`F::grad_checkpoint`]).
    pub fn set_gradient_checkpointing(&self, on: bool) {
        self.gradient_checkpointing.store(on, Ordering::Relaxed);
    }

    /// Freeze the model and attach LoRA adapters to every linear layer
    /// matching `targets` (components of `attn.c_attn`, `attn.c_proj`,
    /// `mlp.c_fc`, `mlp.c_proj`; e.g. `&["c_attn"]` or `&["attn", "mlp"]`).
    /// Returns the number of adapted layers.
    pub fn add_lora(&mut self, rank: usize, alpha: f32, targets: &[&str]) -> usize {
        // freeze everything except adapters added earlier
        for (name, p) in self.named_parameters() {
            if !name.contains(".lora_") {
                p.set_requires_grad(false);
            }
        }
        let mut n = 0;
        for b in &mut self.blocks {
            for (name, lin) in b.linears_mut() {
                if super::llama::lora_target(name, targets) {
                    lin.add_lora(rank, alpha);
                    n += 1;
                }
            }
        }
        assert!(n > 0, "no linear layer matches the LoRA targets {targets:?}");
        n
    }

    /// Fold all LoRA adapters into their weights.
    pub fn merge_lora(&mut self) {
        for b in &mut self.blocks {
            for (_, lin) in b.linears_mut() {
                lin.merge_lora();
            }
        }
    }

    /// Weight-only int8 quantization of every linear layer and of the
    /// (untied) output head, for faster, smaller inference. The token
    /// embedding table stays in its original precision.
    pub fn quantize_int8(&mut self) {
        for b in &mut self.blocks {
            b.attn.c_attn.quantize_int8();
            b.attn.c_proj.quantize_int8();
            b.mlp.c_fc.quantize_int8();
            b.mlp.c_proj.quantize_int8();
        }
        let w = &self.wte.weight;
        let mut head = Linear::from_weight(w.t(), None); // [n_embd, vocab]
        head.quantize_int8();
        self.head_int8 = Some(head);
    }
}

impl Module for GPT2 {
    /// `x`: `[B, T]` token ids -> logits `[B, T, vocab]`.
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_tokens(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("wte", self.wte.named_parameters());
        v.extend(prefixed("wpe", self.wpe.named_parameters()));
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("h.{i}");
            v.extend(prefixed(&format!("{p}.ln_1"), b.ln_1.named_parameters()));
            v.extend(prefixed(&format!("{p}.attn.c_attn"), b.attn.c_attn.named_parameters()));
            v.extend(prefixed(&format!("{p}.attn.c_proj"), b.attn.c_proj.named_parameters()));
            v.extend(prefixed(&format!("{p}.ln_2"), b.ln_2.named_parameters()));
            v.extend(prefixed(&format!("{p}.mlp.c_fc"), b.mlp.c_fc.named_parameters()));
            v.extend(prefixed(&format!("{p}.mlp.c_proj"), b.mlp.c_proj.named_parameters()));
        }
        v.extend(prefixed("ln_f", self.ln_f.named_parameters()));
        v
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("h.{i}");
            for (name, l) in [
                ("attn.c_attn", &b.attn.c_attn),
                ("attn.c_proj", &b.attn.c_proj),
                ("mlp.c_fc", &b.mlp.c_fc),
                ("mlp.c_proj", &b.mlp.c_proj),
            ] {
                v.extend(prefixed(&format!("{p}.{name}"), l.named_buffers()));
            }
        }
        if let Some(h) = &self.head_int8 {
            v.extend(prefixed("lm_head", h.named_buffers()));
        }
        v
    }
}
