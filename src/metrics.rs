//! Small helpers for training loops: running averages, accuracy, throughput.

use crate::ops::IntoIndices;
use crate::tensor::Tensor;
use std::time::Instant;

/// Weighted running mean (e.g. average loss over an epoch).
#[derive(Clone, Debug, Default)]
pub struct RunningMean {
    sum: f64,
    weight: f64,
}

impl RunningMean {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, value: f32) {
        self.add_weighted(value, 1.0);
    }

    /// Add a value counted `weight` times (e.g. a batch mean weighted by batch size).
    pub fn add_weighted(&mut self, value: f32, weight: f32) {
        self.sum += value as f64 * weight as f64;
        self.weight += weight as f64;
    }

    pub fn mean(&self) -> f32 {
        if self.weight == 0.0 {
            0.0
        } else {
            (self.sum / self.weight) as f32
        }
    }

    pub fn count(&self) -> f64 {
        self.weight
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Exponential moving average (smoothed loss curves).
#[derive(Clone, Debug)]
pub struct Ema {
    pub decay: f32,
    value: Option<f32>,
}

impl Ema {
    pub fn new(decay: f32) -> Self {
        Ema { decay, value: None }
    }

    pub fn update(&mut self, x: f32) -> f32 {
        let v = match self.value {
            Some(v) => self.decay * v + (1.0 - self.decay) * x,
            None => x,
        };
        self.value = Some(v);
        v
    }

    pub fn value(&self) -> Option<f32> {
        self.value
    }
}

/// Classification accuracy accumulated over batches.
#[derive(Clone, Debug, Default)]
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count `argmax(logits) == targets` for a `[N, C]` batch.
    pub fn update(&mut self, logits: &Tensor, targets: impl IntoIndices) {
        let pred = logits.argmax(-1, false);
        let t = targets.into_indices(logits.device()).reshape(pred.shape());
        let n = pred.numel();
        self.correct += pred.eq_t(&t).sum_all().item().round() as usize;
        self.total += n;
    }

    pub fn value(&self) -> f32 {
        self.correct as f32 / self.total.max(1) as f32
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Items (samples, tokens, ...) per second since creation or the last reset.
#[derive(Clone, Debug)]
pub struct Throughput {
    start: Instant,
    items: u64,
}

impl Default for Throughput {
    fn default() -> Self {
        Throughput { start: Instant::now(), items: 0 }
    }
}

impl Throughput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, n: usize) {
        self.items += n as u64;
    }

    pub fn per_sec(&self) -> f64 {
        self.items as f64 / self.start.elapsed().as_secs_f64().max(1e-9)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
