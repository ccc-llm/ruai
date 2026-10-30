//! Encoder-decoder Transformer (Vaswani et al. 2017, pre-LN variant) for
//! sequence-to-sequence tasks such as translation: a bidirectional encoder
//! over the source and a causal decoder that also cross-attends to the
//! encoder output ("memory"). Greedy decoding uses a KV cache for the
//! decoder's self-attention.

use super::layers::{Embedding, FeedForward, LayerNorm, Linear};
use super::module::{prefixed, Module};
use super::transformer::{KvCache, LayerKv, MultiHeadAttention, TransformerBlock};
use crate::device::Device;
use crate::functional as F;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seq2SeqConfig {
    pub src_vocab: usize,
    pub tgt_vocab: usize,
    pub dim: usize,
    pub heads: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub ff_hidden: usize,
    /// Maximum source and target length (learned position embeddings).
    pub max_len: usize,
    pub dropout: f32,
}

/// Pre-LN decoder block: causal self-attention, cross-attention, feed-forward.
pub struct DecoderBlock {
    pub ln1: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub ln3: LayerNorm,
    pub ffn: FeedForward,
    pub dropout: f32,
    training: AtomicBool,
}

impl DecoderBlock {
    pub fn new(dim: usize, heads: usize, ff_hidden: usize, dropout: f32, device: Device) -> Self {
        DecoderBlock {
            ln1: LayerNorm::new(dim, device),
            self_attn: MultiHeadAttention::new(dim, heads, dropout, device),
            ln2: LayerNorm::new(dim, device),
            cross_attn: MultiHeadAttention::new(dim, heads, dropout, device),
            ln3: LayerNorm::new(dim, device),
            ffn: FeedForward::new(dim, ff_hidden, device),
            dropout,
            training: AtomicBool::new(true),
        }
    }

    fn drop(&self, x: &Tensor) -> Tensor {
        F::dropout(x, self.dropout, self.training.load(Ordering::Relaxed))
    }

    /// `x` `[B, Tt, D]`, `memory` `[B, Ts, D]`; `self_mask` is usually causal,
    /// `memory_mask` hides source padding (both additive).
    pub fn forward(&self, x: &Tensor, memory: &Tensor, self_mask: Option<&Tensor>, memory_mask: Option<&Tensor>) -> Tensor {
        let h = self.ln1.forward(x);
        let x = x + self.drop(&self.self_attn.attend(&h, &h, self_mask));
        self.cross_and_ffn(&x, memory, memory_mask)
    }

    /// Causal forward of new target positions, extending the self-attention cache.
    pub fn forward_cached(&self, x: &Tensor, memory: &Tensor, memory_mask: Option<&Tensor>, kv: &mut LayerKv) -> Tensor {
        let x = x + self.drop(&self.self_attn.attend_cached(&self.ln1.forward(x), kv));
        self.cross_and_ffn(&x, memory, memory_mask)
    }

    fn cross_and_ffn(&self, x: &Tensor, memory: &Tensor, memory_mask: Option<&Tensor>) -> Tensor {
        let x = x + self.drop(&self.cross_attn.attend(&self.ln2.forward(x), memory, memory_mask));
        &x + self.drop(&self.ffn.forward(&self.ln3.forward(&x)))
    }
}

impl Module for DecoderBlock {
    /// Causal self-attention only (no memory); use [`DecoderBlock::forward`] with a memory.
    fn forward(&self, x: &Tensor) -> Tensor {
        let mask = F::causal_mask(x.dim(1), x.device());
        self.forward(x, x, Some(&mask), None)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("ln1", self.ln1.named_parameters());
        v.extend(prefixed("self_attn", self.self_attn.named_parameters()));
        v.extend(prefixed("ln2", self.ln2.named_parameters()));
        v.extend(prefixed("cross_attn", self.cross_attn.named_parameters()));
        v.extend(prefixed("ln3", self.ln3.named_parameters()));
        v.extend(prefixed("ffn", self.ffn.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        self.self_attn.set_training(training);
        self.cross_attn.set_training(training);
    }
}

/// Encoder-decoder Transformer.
pub struct Seq2Seq {
    pub config: Seq2SeqConfig,
    pub src_emb: Embedding,
    pub src_pos: Embedding,
    pub tgt_emb: Embedding,
    pub tgt_pos: Embedding,
    pub encoder: Vec<TransformerBlock>,
    pub encoder_norm: LayerNorm,
    pub decoder: Vec<DecoderBlock>,
    pub decoder_norm: LayerNorm,
    pub head: Linear,
    training: AtomicBool,
}

fn positions(start: usize, t: usize, device: Device) -> Tensor {
    Tensor::from_indices(&(start..start + t).collect::<Vec<_>>(), &[t], device)
}

impl Seq2Seq {
    pub fn new(config: Seq2SeqConfig, device: Device) -> Self {
        let c = config;
        Seq2Seq {
            src_emb: Embedding::new(c.src_vocab, c.dim, device),
            src_pos: Embedding::new(c.max_len, c.dim, device),
            tgt_emb: Embedding::new(c.tgt_vocab, c.dim, device),
            tgt_pos: Embedding::new(c.max_len, c.dim, device),
            encoder: (0..c.encoder_layers)
                .map(|_| TransformerBlock::new(c.dim, c.heads, c.ff_hidden, c.dropout, false, device))
                .collect(),
            encoder_norm: LayerNorm::new(c.dim, device),
            decoder: (0..c.decoder_layers).map(|_| DecoderBlock::new(c.dim, c.heads, c.ff_hidden, c.dropout, device)).collect(),
            decoder_norm: LayerNorm::new(c.dim, device),
            head: Linear::new(c.dim, c.tgt_vocab, device),
            config,
            training: AtomicBool::new(true),
        }
    }

    fn drop(&self, x: Tensor) -> Tensor {
        F::dropout(&x, self.config.dropout, self.training.load(Ordering::Relaxed))
    }

    /// Encode source tokens `[B, Ts]` into the memory `[B, Ts, D]`.
    /// `src_mask` (see [`F::padding_mask`]) hides source padding.
    pub fn encode(&self, src: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        let t = src.dim(1);
        assert!(t <= self.config.max_len, "source length {t} > max_len {}", self.config.max_len);
        let mut h = self.drop(self.src_emb.forward(src) + self.src_pos.forward(&positions(0, t, src.device())));
        for b in &self.encoder {
            h = b.forward_masked(&h, src_mask);
        }
        self.encoder_norm.forward(&h)
    }

    /// Target logits `[B, Tt, tgt_vocab]` for (teacher-forced) target tokens
    /// `[B, Tt]` given the encoder memory.
    pub fn decode(&self, tgt: &Tensor, memory: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        let t = tgt.dim(1);
        assert!(t <= self.config.max_len, "target length {t} > max_len {}", self.config.max_len);
        let mut h = self.drop(self.tgt_emb.forward(tgt) + self.tgt_pos.forward(&positions(0, t, tgt.device())));
        let causal = F::causal_mask(t, tgt.device());
        for b in &self.decoder {
            h = b.forward(&h, memory, Some(&causal), src_mask);
        }
        self.head.forward(&self.decoder_norm.forward(&h))
    }

    /// Logits for new target tokens after the positions already in `cache`.
    pub fn decode_cached(&self, tgt: &Tensor, memory: &Tensor, src_mask: Option<&Tensor>, cache: &mut KvCache) -> Tensor {
        if cache.layers.len() != self.decoder.len() {
            *cache = KvCache::new(self.decoder.len());
        }
        let (start, t) = (cache.len(), tgt.dim(1));
        assert!(start + t <= self.config.max_len, "target length {} > max_len {}", start + t, self.config.max_len);
        let mut h = self.tgt_emb.forward(tgt) + self.tgt_pos.forward(&positions(start, t, tgt.device()));
        for (b, kv) in self.decoder.iter().zip(cache.layers.iter_mut()) {
            h = b.forward_cached(&h, memory, src_mask, kv);
        }
        self.head.forward(&self.decoder_norm.forward(&h))
    }

    /// Full teacher-forced pass: logits `[B, Tt, tgt_vocab]`.
    pub fn forward_pair(&self, src: &Tensor, tgt: &Tensor, src_mask: Option<&Tensor>) -> Tensor {
        self.decode(tgt, &self.encode(src, src_mask), src_mask)
    }

    /// Greedy decoding of one source sequence: starts from `bos`, stops after
    /// `eos` (if given) or `max_new` tokens; returns the generated tokens
    /// (without `bos`).
    pub fn greedy(&self, src: &[usize], bos: usize, eos: Option<usize>, max_new: usize) -> Vec<usize> {
        let dev = self.head.weight.device();
        let max_new = max_new.min(self.config.max_len);
        crate::no_grad(|| {
            let memory = self.encode(&Tensor::from_indices(src, &[1, src.len()], dev), None);
            let mut cache = KvCache::new(self.decoder.len());
            let mut out = Vec::new();
            let mut last = bos;
            for _ in 0..max_new {
                let logits = self.decode_cached(&Tensor::from_indices(&[last], &[1, 1], dev), &memory, None, &mut cache);
                last = crate::generation::argmax(&logits.to_vec());
                out.push(last);
                if Some(last) == eos {
                    break;
                }
            }
            out
        })
    }
}

impl Module for Seq2Seq {
    /// Runs the encoder only (`[B, Ts]` -> memory); see [`Seq2Seq::forward_pair`].
    fn forward(&self, x: &Tensor) -> Tensor {
        self.encode(x, None)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("src_emb", self.src_emb.named_parameters());
        v.extend(prefixed("src_pos", self.src_pos.named_parameters()));
        v.extend(prefixed("tgt_emb", self.tgt_emb.named_parameters()));
        v.extend(prefixed("tgt_pos", self.tgt_pos.named_parameters()));
        for (i, b) in self.encoder.iter().enumerate() {
            v.extend(prefixed(&format!("encoder.{i}"), b.named_parameters()));
        }
        v.extend(prefixed("encoder_norm", self.encoder_norm.named_parameters()));
        for (i, b) in self.decoder.iter().enumerate() {
            v.extend(prefixed(&format!("decoder.{i}"), b.named_parameters()));
        }
        v.extend(prefixed("decoder_norm", self.decoder_norm.named_parameters()));
        v.extend(prefixed("head", self.head.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        self.encoder.iter().for_each(|b| b.set_training(training));
        self.decoder.iter().for_each(|b| b.set_training(training));
    }
}
