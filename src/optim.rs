//! Optimizers. Updates are computed with ordinary tensor ops under `no_grad`
//! and written back with `set_storage`, so they run on whatever device the
//! parameters live on.

use crate::backend::{AdamParams, DType, Storage};
use crate::tensor::{no_grad, Tensor};

pub trait Optimizer {
    /// Apply one update using the parameters' current `.grad()`.
    fn step(&mut self);
    fn params(&self) -> &[Tensor];
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);

    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
}

/// Stochastic gradient descent with optional momentum and weight decay.
pub struct SGD {
    params: Vec<Tensor>,
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocity: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_momentum(params, lr, 0.0, 0.0)
    }

    pub fn with_momentum(params: Vec<Tensor>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        let n = params.len();
        SGD { params, lr, momentum, weight_decay, velocity: vec![None; n] }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        no_grad(|| {
            for (p, vel) in self.params.iter().zip(self.velocity.iter_mut()) {
                let Some(mut g) = p.grad() else { continue };
                if self.weight_decay != 0.0 {
                    g = g + p * self.weight_decay;
                }
                if self.momentum != 0.0 {
                    let v = match vel.take() {
                        Some(v) => v * self.momentum + &g,
                        None => g,
                    };
                    g = v.clone();
                    *vel = Some(v);
                }
                p.set_storage((p - g * self.lr).storage());
            }
        });
    }
    fn params(&self) -> &[Tensor] {
        &self.params
    }
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam, or AdamW when `decoupled_weight_decay` is set.
pub struct Adam {
    params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub decoupled_weight_decay: bool,
    m: Vec<Option<Storage>>,
    v: Vec<Option<Storage>>,
    t: i32,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
        }
    }

    /// AdamW (Loshchilov & Hutter) with decoupled weight decay.
    pub fn adamw(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        let mut a = Self::new(params, lr);
        a.weight_decay = weight_decay;
        a.decoupled_weight_decay = true;
        a
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let h = AdamParams {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            decoupled: self.decoupled_weight_decay,
            bias_correction1: 1.0 - self.beta1.powi(self.t),
            bias_correction2: 1.0 - self.beta2.powi(self.t),
        };
        for i in 0..self.params.len() {
            let p = &self.params[i];
            let Some(g) = p.grad() else { continue };
            let be = p.backend();
            let zeros = || be.full(p.numel(), 0.0, DType::F32);
            let m = self.m[i].take().unwrap_or_else(zeros);
            let v = self.v[i].take().unwrap_or_else(zeros);
            // one fused kernel per parameter
            let (np, nm, nv) = be.adam_step(&p.storage(), &g.storage(), &m, &v, h);
            p.set_storage(np);
            self.m[i] = Some(nm);
            self.v[i] = Some(nv);
        }
    }
    fn params(&self) -> &[Tensor] {
        &self.params
    }
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Scale all gradients so their global L2 norm is at most `max_norm`.
/// Returns the norm before clipping.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    no_grad(|| {
        let total: f32 = params
            .iter()
            .filter_map(|p| p.grad())
            .map(|g| (&g * &g).sum_all().item())
            .sum::<f32>()
            .sqrt();
        if total > max_norm {
            let scale = max_norm / (total + 1e-6);
            for p in params {
                if let Some(g) = p.grad() {
                    p.set_grad(Some(g * scale));
                }
            }
        }
        total
    })
}

/// Cosine learning-rate schedule with linear warmup.
pub fn cosine_lr(step: usize, total: usize, warmup: usize, base_lr: f32, min_lr: f32) -> f32 {
    if step < warmup {
        return base_lr * (step + 1) as f32 / warmup as f32;
    }
    let progress = (step - warmup) as f32 / (total - warmup).max(1) as f32;
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos())
}
