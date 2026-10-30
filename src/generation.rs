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
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler { temperature, top_k, top_p, rng: Rng::new(seed) }
    }

    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 0)
    }

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
