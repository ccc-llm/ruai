//! Optimizers. Updates are computed with ordinary tensor ops under `no_grad`
//! and written back with `set_storage`, so they run on whatever device the
//! parameters live on.

use crate::nn::backend::{AdamParams, DType, Storage};
use crate::nn::tensor::{no_grad, Tensor};
use std::collections::HashMap;
use std::io;

pub mod lr;

/// A parameter-update rule over a fixed list of parameters.
pub trait Optimizer {
    /// Apply one update using the parameters' current `.grad()`.
    fn step(&mut self);
    /// The parameters being optimized.
    fn params(&self) -> &[Tensor];
    /// Current learning rate.
    fn lr(&self) -> f32;
    /// Set the learning rate (e.g. from an [`lr::LrScheduler`]).
    fn set_lr(&mut self, lr: f32);

    /// Short identifier stored in checkpoints (e.g. `"adam"`).
    fn name(&self) -> &'static str;

    /// Internal state (moment buffers, step count, learning rate) as named
    /// tensors. Per-parameter entries are keyed by the parameter's index in
    /// `params()`, so the optimizer must be rebuilt over the same parameter
    /// list before loading.
    fn state_dict(&self) -> Vec<(String, Tensor)>;

    /// Restore state produced by [`Optimizer::state_dict`] (tensors are moved
    /// to each parameter's device).
    fn load_state_dict(&mut self, state: &[(String, Tensor)]) -> io::Result<()>;

    /// Clear the gradients of all parameters.
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
}

/// Stochastic gradient descent with optional momentum and weight decay.
pub struct SGD {
    params: Vec<Tensor>,
    /// Learning rate.
    pub lr: f32,
    /// Momentum factor (0 = plain SGD).
    pub momentum: f32,
    /// L2 penalty added to the gradient (0 = none).
    pub weight_decay: f32,
    velocity: Vec<Option<Tensor>>,
}

impl SGD {
    /// Plain SGD without momentum or weight decay.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_momentum(params, lr, 0.0, 0.0)
    }

    /// SGD with `momentum` and L2 `weight_decay`.
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
    fn name(&self) -> &'static str {
        "sgd"
    }
    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("lr".to_string(), Tensor::scalar(self.lr, crate::nn::Device::Cpu))];
        for (i, v) in self.velocity.iter().enumerate() {
            if let Some(v) = v {
                out.push((format!("velocity.{i}"), v.clone()));
            }
        }
        out
    }
    fn load_state_dict(&mut self, state: &[(String, Tensor)]) -> io::Result<()> {
        let map = StateMap::new(state);
        self.lr = map.scalar("lr")?;
        for (i, p) in self.params.iter().enumerate() {
            self.velocity[i] = map.per_param("velocity", i, p)?.map(|s| Tensor::from_storage(s, p.shape().to_vec()));
        }
        map.check_indices(&["velocity"], self.params.len())
    }
}

/// Named-state lookup shared by the optimizers' `load_state_dict`.
struct StateMap<'a>(HashMap<&'a str, &'a Tensor>);

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl<'a> StateMap<'a> {
    fn new(state: &'a [(String, Tensor)]) -> Self {
        StateMap(state.iter().map(|(n, t)| (n.as_str(), t)).collect())
    }

    fn scalar(&self, key: &str) -> io::Result<f32> {
        self.0.get(key).map(|t| t.to_vec()[0]).ok_or_else(|| bad(format!("optimizer state is missing '{key}'")))
    }

    /// `prefix.i` moved to `p`'s device, checked against `p`'s shape.
    fn per_param(&self, prefix: &str, i: usize, p: &Tensor) -> io::Result<Option<Storage>> {
        let Some(t) = self.0.get(format!("{prefix}.{i}").as_str()) else { return Ok(None) };
        if t.shape() != p.shape() {
            return Err(bad(format!("optimizer state '{prefix}.{i}' has shape {:?}, parameter has {:?}", t.shape(), p.shape())));
        }
        Ok(Some(t.storage().to_device(p.device())))
    }

    /// Reject per-parameter entries beyond the parameter list (a different model).
    fn check_indices(&self, prefixes: &[&str], n: usize) -> io::Result<()> {
        for key in self.0.keys() {
            for pre in prefixes {
                if let Some(i) = key.strip_prefix(pre).and_then(|r| r.strip_prefix('.')).and_then(|r| r.parse::<usize>().ok()) {
                    if i >= n {
                        return Err(bad(format!("optimizer state '{key}' refers to parameter {i}, but there are only {n}")));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Adam, or AdamW when `decoupled_weight_decay` is set.
pub struct Adam {
    params: Vec<Tensor>,
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay (default 0.9).
    pub beta1: f32,
    /// Second-moment decay (default 0.999).
    pub beta2: f32,
    /// Denominator epsilon (default 1e-8).
    pub eps: f32,
    /// Weight-decay coefficient (default 0).
    pub weight_decay: f32,
    /// Apply weight decay directly to the weights (AdamW) instead of the gradient.
    pub decoupled_weight_decay: bool,
    m: Vec<Option<Storage>>,
    v: Vec<Option<Storage>>,
    t: i32,
}

impl Adam {
    /// Adam with the default betas `(0.9, 0.999)`, `eps = 1e-8` and no weight decay.
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
    fn name(&self) -> &'static str {
        if self.decoupled_weight_decay {
            "adamw"
        } else {
            "adam"
        }
    }
    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let cpu = crate::nn::Device::Cpu;
        let mut out = vec![
            ("lr".to_string(), Tensor::scalar(self.lr, cpu)),
            ("step".to_string(), Tensor::from_vec_i32(vec![self.t], &[], cpu)),
        ];
        for (i, p) in self.params.iter().enumerate() {
            for (key, s) in [("m", &self.m[i]), ("v", &self.v[i])] {
                if let Some(s) = s {
                    out.push((format!("{key}.{i}"), Tensor::from_storage(s.clone(), p.shape().to_vec())));
                }
            }
        }
        out
    }
    fn load_state_dict(&mut self, state: &[(String, Tensor)]) -> io::Result<()> {
        let map = StateMap::new(state);
        self.lr = map.scalar("lr")?;
        self.t = map.scalar("step")? as i32;
        for (i, p) in self.params.iter().enumerate() {
            self.m[i] = map.per_param("m", i, p)?;
            self.v[i] = map.per_param("v", i, p)?;
        }
        map.check_indices(&["m", "v"], self.params.len())
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

/// Cosine learning-rate schedule with linear warmup
/// (shorthand for [`lr::CosineLr`]).
pub fn cosine_lr(step: usize, total: usize, warmup: usize, base_lr: f32, min_lr: f32) -> f32 {
    use lr::LrScheduler;
    lr::CosineLr { base_lr, min_lr, warmup, total }.lr_at(step)
}
