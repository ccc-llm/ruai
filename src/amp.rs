//! Automatic mixed precision: [`autocast`] regions plus a dynamic
//! [`GradScaler`] for F16 (which has a narrow exponent range; BF16 usually
//! trains without scaling).
//!
//! ```no_run
//! use ruai::amp::{autocast, GradScaler};
//! # use ruai::{nn::{Linear, Module}, optim::{Adam, Optimizer}, functional as F, DType, Device, Tensor};
//! # let model = Linear::new(4, 2, Device::Mps);
//! # let mut opt = Adam::new(model.parameters(), 1e-3);
//! # let (x, y) = (Tensor::randn(&[8, 4], Device::Mps), vec![0usize; 8]);
//! let mut scaler = GradScaler::new();
//! opt.zero_grad();
//! let loss = autocast(DType::F16, || F::cross_entropy(&model.forward(&x), &y, 0.0));
//! scaler.scale(&loss).backward();
//! scaler.unscale(&opt);               // optional: needed before clip_grad_norm
//! scaler.step(&mut opt);              // skips the step if gradients overflowed
//! ```

pub use crate::tensor::{autocast, autocast_dtype, AutocastGuard};

use crate::optim::Optimizer;
use crate::serialize::Metadata;
use crate::tensor::{no_grad, Tensor};

/// Dynamic loss scaling (like `torch.cuda.amp.GradScaler`).
///
/// The loss is multiplied by `scale` before `backward()` so small F16
/// gradients do not flush to zero. Before the optimizer step the gradients
/// are divided by `scale`; if any is inf/NaN the step is skipped and the
/// scale multiplied by `backoff_factor` (default 0.5); after
/// `growth_interval` clean steps it is multiplied by `growth_factor` (default 2).
#[derive(Clone, Debug)]
pub struct GradScaler {
    scale: f32,
    /// Scale multiplier after `growth_interval` clean steps (default 2).
    pub growth_factor: f32,
    /// Scale multiplier after an overflow (default 0.5).
    pub backoff_factor: f32,
    /// Consecutive clean steps before growing the scale (default 2000).
    pub growth_interval: u32,
    good_steps: u32,
    /// `Some(found_inf)` once `unscale` ran for the current step.
    unscaled: Option<bool>,
    skipped: u64,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl GradScaler {
    /// A scaler with initial scale `65536` (2^16).
    pub fn new() -> Self {
        Self::with_scale(65536.0)
    }

    /// A scaler with the given initial scale.
    pub fn with_scale(scale: f32) -> Self {
        GradScaler {
            scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            good_steps: 0,
            unscaled: None,
            skipped: 0,
        }
    }

    /// Current loss scale.
    pub fn scale_value(&self) -> f32 {
        self.scale
    }

    /// Number of optimizer steps skipped because of inf/NaN gradients.
    pub fn skipped_steps(&self) -> u64 {
        self.skipped
    }

    /// `loss * scale` (call `.backward()` on the result).
    pub fn scale(&self, loss: &Tensor) -> Tensor {
        loss * self.scale
    }

    /// Divide the optimizer's gradients by the scale in place and check them.
    /// Returns `true` if all gradients are finite. Idempotent within a step.
    pub fn unscale(&mut self, opt: &dyn Optimizer) -> bool {
        if let Some(found_inf) = self.unscaled {
            return !found_inf;
        }
        let inv = 1.0 / self.scale;
        let found_inf = no_grad(|| {
            // Any inf/NaN makes a sum non-finite (inf - inf = NaN). The sums
            // stay on the device and are combined, so the whole check costs a
            // single host synchronization instead of one per parameter.
            let mut total: Option<Tensor> = None;
            for p in opt.params() {
                if let Some(g) = p.grad() {
                    let g = g * inv;
                    let s = g.sum_all().to_dtype(crate::DType::F32);
                    total = Some(match total {
                        Some(t) => t + s,
                        None => s,
                    });
                    p.set_grad(Some(g));
                }
            }
            total.is_some_and(|t| !t.item().is_finite())
        });
        self.unscaled = Some(found_inf);
        !found_inf
    }

    /// Unscale (if not done yet), then step the optimizer unless gradients
    /// overflowed, and update the scale. Returns whether the step was taken.
    pub fn step(&mut self, opt: &mut dyn Optimizer) -> bool {
        let finite = self.unscale(opt);
        self.unscaled = None;
        if finite {
            opt.step();
            self.good_steps += 1;
            if self.good_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.good_steps = 0;
            }
        } else {
            opt.zero_grad();
            self.scale *= self.backoff_factor;
            self.good_steps = 0;
            self.skipped += 1;
        }
        finite
    }

    /// Scaler state as checkpoint metadata (merge into `TrainState::extra`).
    pub fn state(&self) -> Metadata {
        let mut m = Metadata::new();
        m.insert("amp.scale".into(), self.scale.to_string());
        m.insert("amp.good_steps".into(), self.good_steps.to_string());
        m.insert("amp.skipped".into(), self.skipped.to_string());
        m
    }

    /// Restore from metadata written by [`GradScaler::state`] (missing keys keep defaults).
    pub fn load_state(&mut self, m: &Metadata) {
        if let Some(v) = m.get("amp.scale").and_then(|v| v.parse().ok()) {
            self.scale = v;
        }
        if let Some(v) = m.get("amp.good_steps").and_then(|v| v.parse().ok()) {
            self.good_steps = v;
        }
        if let Some(v) = m.get("amp.skipped").and_then(|v| v.parse().ok()) {
            self.skipped = v;
        }
    }
}
