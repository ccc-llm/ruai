//! Learning-rate schedules: pure functions of the step number, so they need
//! no state of their own and resume correctly from a checkpointed step.
//!
//! ```
//! use ruai::optim::lr::{CosineLr, LrScheduler};
//! let s = CosineLr { base_lr: 1e-3, min_lr: 1e-4, warmup: 10, total: 100 };
//! assert!((s.lr_at(9) - 1e-3).abs() < 1e-9);
//! assert!((s.lr_at(100) - 1e-4).abs() < 1e-9);
//! ```

use super::Optimizer;
use std::f32::consts::PI;

/// A learning-rate schedule: a function of the step number.
pub trait LrScheduler {
    /// Learning rate for (0-based) training step `step`.
    fn lr_at(&self, step: usize) -> f32;

    /// Set the optimizer's learning rate for `step`; returns it.
    fn apply(&self, opt: &mut dyn Optimizer, step: usize) -> f32 {
        let lr = self.lr_at(step);
        opt.set_lr(lr);
        lr
    }
}

/// Linear ramp from ~0 to 1 over `warmup` steps.
fn warmup_factor(step: usize, warmup: usize) -> f32 {
    if warmup == 0 {
        1.0
    } else {
        ((step + 1) as f32 / warmup as f32).min(1.0)
    }
}

/// A fixed learning rate.
#[derive(Clone, Copy, Debug)]
pub struct ConstantLr(pub f32);

impl LrScheduler for ConstantLr {
    fn lr_at(&self, _step: usize) -> f32 {
        self.0
    }
}

/// Multiply by `gamma` every `step_size` steps.
#[derive(Clone, Copy, Debug)]
pub struct StepLr {
    /// Learning rate at step 0.
    pub base_lr: f32,
    /// Steps between decays.
    pub step_size: usize,
    /// Decay factor applied every `step_size` steps.
    pub gamma: f32,
}

impl LrScheduler for StepLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi((step / self.step_size.max(1)) as i32)
    }
}

/// Multiply by `gamma` every step.
#[derive(Clone, Copy, Debug)]
pub struct ExponentialLr {
    /// Learning rate at step 0.
    pub base_lr: f32,
    /// Decay factor per step.
    pub gamma: f32,
}

impl LrScheduler for ExponentialLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi(step as i32)
    }
}

/// Linear warmup, then cosine decay from `base_lr` to `min_lr` at `total`.
#[derive(Clone, Copy, Debug)]
pub struct CosineLr {
    /// Peak learning rate (reached at the end of warmup).
    pub base_lr: f32,
    /// Final learning rate (at and after `total`).
    pub min_lr: f32,
    /// Number of linear warmup steps (0 = none).
    pub warmup: usize,
    /// Step at which decay reaches `min_lr`.
    pub total: usize,
}

impl LrScheduler for CosineLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup {
            return self.base_lr * warmup_factor(step, self.warmup);
        }
        let progress = ((step - self.warmup) as f32 / (self.total.saturating_sub(self.warmup)).max(1) as f32).min(1.0);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }
}

/// Linear warmup, then linear decay from `base_lr` to `min_lr` at `total`.
#[derive(Clone, Copy, Debug)]
pub struct LinearLr {
    /// Peak learning rate (reached at the end of warmup).
    pub base_lr: f32,
    /// Final learning rate (at and after `total`).
    pub min_lr: f32,
    /// Number of linear warmup steps (0 = none).
    pub warmup: usize,
    /// Step at which decay reaches `min_lr`.
    pub total: usize,
}

impl LrScheduler for LinearLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup {
            return self.base_lr * warmup_factor(step, self.warmup);
        }
        let progress = ((step - self.warmup) as f32 / (self.total.saturating_sub(self.warmup)).max(1) as f32).min(1.0);
        self.base_lr + (self.min_lr - self.base_lr) * progress
    }
}
