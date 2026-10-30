//! Token sampling for autoregressive generation.

use crate::rand::Rng;

/// Index of the largest logit.
pub fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

/// Picks the next token from a logit vector.
///
/// * `temperature <= 0`: greedy (argmax)
/// * `top_k > 0`: only the `k` most likely tokens are considered
/// * `top_p < 1`: nucleus sampling over the smallest set with mass >= `p`
#[derive(Clone, Debug)]
pub struct Sampler {
    /// Softmax temperature; `<= 0` means greedy.
    pub temperature: f32,
    /// Keep only the `top_k` highest logits (0 = no limit).
    pub top_k: usize,
    /// Nucleus threshold on cumulative probability (1 = off).
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    /// A sampler with its own RNG seeded by `seed`.
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler { temperature, top_k, top_p, rng: Rng::new(seed) }
    }

    /// Always pick the argmax.
    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 0)
    }

    /// Draw a token index from `logits` (length `vocab`).
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        // candidates sorted by logit, descending
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.top_k > 0 {
            idx.truncate(self.top_k.min(idx.len()));
        }
        let m = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - m) / self.temperature).exp()).collect();
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
        if self.top_p < 1.0 {
            let mut acc = 0.0;
            let keep = probs
                .iter()
                .position(|&p| {
                    acc += p;
                    acc >= self.top_p
                })
                .map_or(probs.len(), |i| i + 1);
            probs.truncate(keep);
            idx.truncate(keep);
        }
        let r = self.rng.unit() * probs.iter().sum::<f32>();
        let mut acc = 0.0;
        for (p, &i) in probs.iter().zip(&idx) {
            acc += p;
            if r < acc {
                return i;
            }
        }
        *idx.last().unwrap()
    }
}

/// Autoregressive decoding with a KV cache, shared by the language models.
///
/// `step(tokens [1, T], cache)` returns logits `[1, T, vocab]` for the new
/// tokens and extends `cache`. Past `max_len` tokens the context window
/// slides over the most recent `max_len` tokens (re-encoded from an empty
/// cache). `on_token` is called after each new token; returning `false`
/// stops early. Returns the prompt followed by the generated tokens.
pub fn generate_cached(
    prompt: &[usize],
    n: usize,
    max_len: usize,
    device: crate::Device,
    sampler: &mut Sampler,
    mut step: impl FnMut(&crate::Tensor, &mut crate::nn::KvCache) -> crate::Tensor,
    mut on_token: impl FnMut(usize) -> bool,
) -> Vec<usize> {
    assert!(!prompt.is_empty(), "prompt must not be empty");
    let mut ids = prompt.to_vec();
    let mut cache = crate::nn::KvCache::default();
    crate::no_grad(|| {
        let mut pending = ids[ids.len().saturating_sub(max_len)..].to_vec();
        for _ in 0..n {
            if cache.len() + pending.len() > max_len {
                cache.clear();
                pending = ids[ids.len() - max_len..].to_vec();
            }
            let logits = step(&crate::Tensor::from_indices(&pending, &[1, pending.len()], device), &mut cache);
            let v = logits.dim(-1);
            let next = sampler.sample(&logits.narrow(1, pending.len() - 1, 1).reshape(&[v]).to_vec());
            ids.push(next);
            pending = vec![next];
            if !on_token(next) {
                break;
            }
        }
    });
    ids
}
