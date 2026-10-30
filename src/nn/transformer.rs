use super::layers::{FeedForward, LayerNorm, Linear};
use super::module::{prefixed, Module};
use crate::device::Device;
use crate::functional as F;
use crate::generation::Sampler;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cached keys and values of one attention layer, `[B, H, T_past, D/H]`.
#[derive(Clone, Default)]
pub struct LayerKv {
    /// Keys `[B, H, T_past, D/H]`.
    pub k: Option<Tensor>,
    /// Values `[B, H, T_past, D/H]`.
    pub v: Option<Tensor>,
}

impl LayerKv {
    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dim(2))
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append new keys/values `[B, H, T_new, D/H]`; returns all of them.
    pub fn append(&mut self, k: Tensor, v: Tensor) -> (Tensor, Tensor) {
        let (k, v) = match (self.k.take(), self.v.take()) {
            (Some(pk), Some(pv)) => (Tensor::cat(&[&pk, &k], 2), Tensor::cat(&[&pv, &v], 2)),
            _ => (k, v),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        (k, v)
    }
}

/// Key/value cache for incremental (token-by-token) decoding: each new token
/// only computes its own query/key/value and attends to the cached ones, so
/// generating `n` tokens costs O(n) layer evaluations instead of O(n²).
#[derive(Clone, Default)]
pub struct KvCache {
    /// One entry per attention layer.
    pub layers: Vec<LayerKv>,
}

impl KvCache {
    /// An empty cache for `n_layers` layers.
    pub fn new(n_layers: usize) -> Self {
        KvCache { layers: vec![LayerKv::default(); n_layers] }
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, LayerKv::len)
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all cached positions.
    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = LayerKv::default());
    }
}

/// Multi-head attention over `[B, T, D]` inputs.
pub struct MultiHeadAttention {
    /// Query projection.
    pub wq: Linear,
    /// Key projection.
    pub wk: Linear,
    /// Value projection.
    pub wv: Linear,
    /// Output projection.
    pub wo: Linear,
    /// Number of heads.
    pub n_heads: usize,
    /// Dropout on the attention weights (training only).
    pub dropout: f32,
    training: AtomicBool,
}

impl MultiHeadAttention {
    /// Attention of width `dim` split into `n_heads` heads.
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

    /// Causal self-attention of new positions `x` `[B, T_new, D]` against
    /// the cached ones plus themselves; appends the new keys/values to `kv`.
    pub fn attend_cached(&self, x: &Tensor, kv: &mut LayerKv) -> Tensor {
        let (b, tq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let past = kv.len();
        let q = self.split_heads(&self.wq.forward(x));
        let (k, v) = kv.append(self.split_heads(&self.wk.forward(x)), self.split_heads(&self.wv.forward(x)));
        // a single new query may see every key; otherwise mask the future
        let mask = (tq > 1).then(|| F::causal_mask_offset(tq, past, x.device()));
        let training = self.training.load(Ordering::Relaxed);
        let o = F::scaled_dot_product_attention(&q, &k, &v, mask.as_ref(), self.dropout, training);
        self.wo.forward(&o.permute(&[0, 2, 1, 3]).reshape(&[b, tq, d]))
    }
}

impl Clone for MultiHeadAttention {
    /// Shares the parameters (like cloning a [`Tensor`]).
    fn clone(&self) -> Self {
        MultiHeadAttention {
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            n_heads: self.n_heads,
            dropout: self.dropout,
            training: AtomicBool::new(self.training.load(Ordering::Relaxed)),
        }
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
    /// Norm before attention.
    pub ln1: LayerNorm,
    /// Self-attention.
    pub attn: MultiHeadAttention,
    /// Norm before the feed-forward.
    pub ln2: LayerNorm,
    /// Position-wise MLP.
    pub ffn: FeedForward,
    /// Whether [`Module::forward`] applies a causal mask.
    pub causal: bool,
    /// Dropout on each residual branch.
    pub dropout: f32,
    training: AtomicBool,
}

impl TransformerBlock {
    /// A block of width `dim` with an MLP hidden size `ff_hidden`.
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

    /// Causal forward of new positions using (and extending) a KV cache.
    pub fn forward_cached(&self, x: &Tensor, kv: &mut LayerKv) -> Tensor {
        let training = self.training.load(Ordering::Relaxed);
        let x = x + F::dropout(&self.attn.attend_cached(&self.ln1.forward(x), kv), self.dropout, training);
        let h = self.ln2.forward(&x);
        &x + F::dropout(&self.ffn.forward(&h), self.dropout, training)
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

impl Clone for TransformerBlock {
    /// Shares the parameters (like cloning a [`Tensor`]).
    fn clone(&self) -> Self {
        TransformerBlock {
            ln1: self.ln1.clone(),
            attn: self.attn.clone(),
            ln2: self.ln2.clone(),
            ffn: self.ffn.clone(),
            causal: self.causal,
            dropout: self.dropout,
            training: AtomicBool::new(self.training.load(Ordering::Relaxed)),
        }
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
    /// Token embeddings.
    pub tok_emb: super::layers::Embedding,
    /// Learned position embeddings.
    pub pos_emb: super::layers::Embedding,
    /// Causal transformer blocks.
    pub blocks: Vec<TransformerBlock>,
    /// Final layer norm.
    pub ln_f: LayerNorm,
    /// Output projection to vocabulary logits.
    pub head: Linear,
    /// Maximum context length.
    pub max_len: usize,
    gradient_checkpointing: AtomicBool,
}

/// Hyperparameters of [`GPT`].
#[derive(Clone, Copy, Debug)]
pub struct GPTConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub max_len: usize,
    /// Model width.
    pub dim: usize,
    /// Attention heads per block.
    pub n_heads: usize,
    /// Number of blocks.
    pub n_layers: usize,
    /// Dropout probability (training only).
    pub dropout: f32,
}

impl GPT {
    /// Randomly initialized model.
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
            gradient_checkpointing: AtomicBool::new(false),
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
        let mut x = self.embed(tokens, 0);
        let mask = F::causal_mask(t, x.device());
        let checkpoint = self.gradient_checkpointing.load(Ordering::Relaxed) && crate::is_grad_enabled();
        for blk in &self.blocks {
            x = if checkpoint {
                let b = blk.clone();
                F::grad_checkpoint(&[&x, &mask], blk.parameters(), move |xs| b.forward_masked(&xs[0], Some(&xs[1])))
            } else {
                blk.forward_masked(&x, Some(&mask))
            };
        }
        self.head.forward(&self.ln_f.forward(&x))
    }

    /// Recompute each block's activations during the backward pass instead
    /// of storing them ([`F::grad_checkpoint`]).
    pub fn set_gradient_checkpointing(&self, on: bool) {
        self.gradient_checkpointing.store(on, Ordering::Relaxed);
    }

    /// Token + position embeddings for positions `start..start + T`.
    fn embed(&self, tokens: &Tensor, start: usize) -> Tensor {
        let t = tokens.shape()[1];
        let pos = Tensor::from_indices(&(start..start + t).collect::<Vec<_>>(), &[t], tokens.device());
        self.tok_emb.forward(tokens) + self.pos_emb.forward(&pos)
    }

    /// Logits `[B, T_new, vocab]` for new tokens `[B, T_new]` following the
    /// `cache.len()` positions already in `cache` (which is extended).
    pub fn forward_cached(&self, tokens: &Tensor, cache: &mut KvCache) -> Tensor {
        if cache.layers.len() != self.blocks.len() {
            *cache = KvCache::new(self.blocks.len());
        }
        let start = cache.len();
        assert!(start + tokens.shape()[1] <= self.max_len, "KV cache would exceed max_len {}", self.max_len);
        let mut x = self.embed(tokens, start);
        for (blk, kv) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            x = blk.forward_cached(&x, kv);
        }
        self.head.forward(&self.ln_f.forward(&x))
    }

    /// Autoregressive generation of `n` new tokens after `prompt` with a KV
    /// cache, picking each token with `sampler` (e.g. [`Sampler::greedy`]).
    /// Past `max_len` tokens the context window slides over the last
    /// `max_len` tokens (re-encoded without the cache), like
    /// [`GPT::generate_uncached`]. Returns the prompt followed by the new tokens.
    pub fn generate(&self, prompt: &[usize], n: usize, sampler: &mut Sampler) -> Vec<usize> {
        let dev = self.head.weight.device();
        crate::generation::generate_cached(prompt, n, self.max_len, dev, sampler, |t, c| self.forward_cached(t, c), |_| true)
    }

    /// Like [`GPT::generate`] but recomputes the whole context every step
    /// (no KV cache); for reference and testing.
    pub fn generate_uncached(&self, prompt: &[usize], n: usize) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        crate::no_grad(|| {
            for _ in 0..n {
                let start = ids.len().saturating_sub(self.max_len);
                let ctx = &ids[start..];
                let logits = self.forward_ids(ctx, 1, ctx.len());
                let v = logits.dim(-1);
                let last = logits.reshape(&[ctx.len(), v]).to_vec()[(ctx.len() - 1) * v..].to_vec();
                ids.push(crate::generation::argmax(&last));
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
