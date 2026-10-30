use super::layers::{FeedForward, LayerNorm, Linear};
use super::module::{prefixed, Module};
use crate::device::Device;
use crate::functional as F;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Multi-head attention over `[B, T, D]` inputs.
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub dropout: f32,
    training: AtomicBool,
}

impl MultiHeadAttention {
    pub fn new(dim: usize, n_heads: usize, dropout: f32, device: Device) -> Self {
        assert_eq!(dim % n_heads, 0, "dim {dim} must be divisible by n_heads {n_heads}");
        MultiHeadAttention {
            wq: Linear::new(dim, dim, device),
            wk: Linear::new(dim, dim, device),
            wv: Linear::new(dim, dim, device),
            wo: Linear::new(dim, dim, device),
            n_heads,
            dropout,
            training: AtomicBool::new(true),
        }
    }

    /// `[B, T, D]` -> `[B, H, T, D/H]`
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        x.reshape(&[b, t, self.n_heads, d / self.n_heads]).permute(&[0, 2, 1, 3])
    }

    /// General attention: queries from `xq` `[B, Tq, D]`, keys/values from
    /// `xkv` `[B, Tk, D]` (pass the same tensor for self-attention). `mask` is
    /// additive and broadcastable to `[B, H, Tq, Tk]`.
    pub fn attend(&self, xq: &Tensor, xkv: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (b, tq, d) = (xq.shape()[0], xq.shape()[1], xq.shape()[2]);
        let q = self.split_heads(&self.wq.forward(xq));
        let k = self.split_heads(&self.wk.forward(xkv));
        let v = self.split_heads(&self.wv.forward(xkv));
        let training = self.training.load(Ordering::Relaxed);
        let o = F::scaled_dot_product_attention(&q, &k, &v, mask, self.dropout, training);
        let o = o.permute(&[0, 2, 1, 3]).reshape(&[b, tq, d]);
        self.wo.forward(&o)
    }
}

impl Module for MultiHeadAttention {
    /// Unmasked self-attention.
    fn forward(&self, x: &Tensor) -> Tensor {
        self.attend(x, x, None)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("wq", self.wq.named_parameters());
        v.extend(prefixed("wk", self.wk.named_parameters()));
        v.extend(prefixed("wv", self.wv.named_parameters()));
        v.extend(prefixed("wo", self.wo.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}

/// Pre-LayerNorm transformer block:
/// `x = x + Attn(LN(x)); x = x + FFN(LN(x))`.
/// With `causal = true` it is a GPT-style decoder block.
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ffn: FeedForward,
    pub causal: bool,
    pub dropout: f32,
    training: AtomicBool,
}

impl TransformerBlock {
    pub fn new(dim: usize, n_heads: usize, ff_hidden: usize, dropout: f32, causal: bool, device: Device) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(dim, device),
            attn: MultiHeadAttention::new(dim, n_heads, dropout, device),
            ln2: LayerNorm::new(dim, device),
            ffn: FeedForward::new(dim, ff_hidden, device),
            causal,
            dropout,
            training: AtomicBool::new(true),
        }
    }

    /// Forward with an explicit additive attention mask (overrides `causal`).
    pub fn forward_masked(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let training = self.training.load(Ordering::Relaxed);
        let h = self.ln1.forward(x);
        let x = x + F::dropout(&self.attn.attend(&h, &h, mask), self.dropout, training);
        let h = self.ln2.forward(&x);
        &x + F::dropout(&self.ffn.forward(&h), self.dropout, training)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.causal {
            let mask = F::causal_mask(x.shape()[1], x.device());
            self.forward_masked(x, Some(&mask))
        } else {
            self.forward_masked(x, None)
        }
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("ln1", self.ln1.named_parameters());
        v.extend(prefixed("attn", self.attn.named_parameters()));
        v.extend(prefixed("ln2", self.ln2.named_parameters()));
        v.extend(prefixed("ffn", self.ffn.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        self.attn.set_training(training);
    }
}

/// A GPT-style decoder-only language model.
pub struct GPT {
    pub tok_emb: super::layers::Embedding,
    pub pos_emb: super::layers::Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub head: Linear,
    pub max_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub max_len: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dropout: f32,
}

impl GPT {
    pub fn new(cfg: GPTConfig, device: Device) -> Self {
        GPT {
            tok_emb: super::layers::Embedding::new(cfg.vocab_size, cfg.dim, device),
            pos_emb: super::layers::Embedding::new(cfg.max_len, cfg.dim, device),
            blocks: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(cfg.dim, cfg.n_heads, 4 * cfg.dim, cfg.dropout, true, device))
                .collect(),
            ln_f: LayerNorm::new(cfg.dim, device),
            head: Linear::new(cfg.dim, cfg.vocab_size, device),
            max_len: cfg.max_len,
        }
    }

    /// `ids`: `[B, T]` row-major token ids -> logits `[B, T, vocab]`.
    pub fn forward_ids(&self, ids: &[usize], b: usize, t: usize) -> Tensor {
        let dev = self.head.weight.device();
        self.forward_tokens(&Tensor::from_indices(ids, &[b, t], dev))
    }

    /// `tokens`: `[B, T]` integer tensor -> logits `[B, T, vocab]`.
    pub fn forward_tokens(&self, tokens: &Tensor) -> Tensor {
        let t = tokens.shape()[1];
        assert!(t <= self.max_len, "sequence length {t} > max_len {}", self.max_len);
        let tok = self.tok_emb.forward(tokens);
        let pos = self.pos_emb.forward(&Tensor::arange(t, tokens.device()).to_dtype(crate::DType::I32));
        let mut x = tok + pos;
        let mask = F::causal_mask(t, x.device());
        for blk in &self.blocks {
            x = blk.forward_masked(&x, Some(&mask));
        }
        self.head.forward(&self.ln_f.forward(&x))
    }

    /// Autoregressive sampling of `n` new tokens after `prompt`.
    /// `temperature <= 0` means greedy decoding.
    pub fn generate(&self, prompt: &[usize], n: usize, temperature: f32) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        crate::no_grad(|| {
            for _ in 0..n {
                let start = ids.len().saturating_sub(self.max_len);
                let ctx = &ids[start..];
                let logits = self.forward_ids(ctx, 1, ctx.len());
                let v = logits.dim(-1);
                let last = logits.reshape(&[ctx.len(), v]).to_vec()[(ctx.len() - 1) * v..].to_vec();
                let next = if temperature <= 0.0 {
                    last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
                } else {
                    let m = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let p: Vec<f32> = last.iter().map(|&l| ((l - m) / temperature).exp()).collect();
                    crate::rand::sample_categorical(&p)
                };
                ids.push(next);
            }
        });
        ids
    }
}

impl Module for GPT {
    /// `x`: `[B, T]` token ids.
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_tokens(x)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("tok_emb", self.tok_emb.named_parameters());
        v.extend(prefixed("pos_emb", self.pos_emb.named_parameters()));
        for (i, b) in self.blocks.iter().enumerate() {
            v.extend(prefixed(&format!("blocks.{i}"), b.named_parameters()));
        }
        v.extend(prefixed("ln_f", self.ln_f.named_parameters()));
        v.extend(prefixed("head", self.head.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.blocks.iter().for_each(|b| b.set_training(training));
    }
}
