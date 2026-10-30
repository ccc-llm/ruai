use super::module::{prefixed, Module};
use crate::device::Device;
use crate::functional as F;
use crate::tensor::{no_grad, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Uniform(-bound, bound) parameter, the PyTorch default for Linear/Conv.
fn uniform_param(shape: &[usize], bound: f32, device: Device) -> Tensor {
    Tensor::rand_uniform(shape, -bound, bound, device).requires_grad_()
}

/// Fully connected layer `y = x @ W + b`, with `W` stored as `[in, out]`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, device: Device) -> Self {
        Self::with_bias(in_features, out_features, true, device)
    }

    pub fn with_bias(in_features: usize, out_features: usize, bias: bool, device: Device) -> Self {
        let bound = 1.0 / (in_features as f32).sqrt();
        Linear {
            weight: uniform_param(&[in_features, out_features], bound, device),
            bias: bias.then(|| uniform_param(&[out_features], bound, device)),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.matmul(&self.weight);
        match &self.bias {
            Some(b) => y + b,
            None => y,
        }
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            v.push(("bias".to_string(), b.clone()));
        }
        v
    }
}

/// 2D convolution over `[N, C, H, W]` with a square kernel.
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, device: Device) -> Self {
        let bound = 1.0 / ((in_ch * kernel * kernel) as f32).sqrt();
        Conv2d {
            weight: uniform_param(&[out_ch, in_ch, kernel, kernel], bound, device),
            bias: Some(uniform_param(&[out_ch], bound, device)),
            stride,
            padding,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::conv2d(x, &self.weight, self.bias.as_ref(), self.stride, self.padding)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            v.push(("bias".to_string(), b.clone()));
        }
        v
    }
}

pub struct MaxPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel: usize, stride: usize) -> Self {
        MaxPool2d { kernel, stride }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::max_pool2d(x, self.kernel, self.stride)
    }
}

pub struct AvgPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel: usize, stride: usize) -> Self {
        AvgPool2d { kernel, stride }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::avg_pool2d(x, self.kernel, self.stride)
    }
}

/// Batch normalization over the channel dim of `[N, C, H, W]`.
pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub momentum: f32,
    pub eps: f32,
    training: AtomicBool,
}

impl BatchNorm2d {
    pub fn new(channels: usize, device: Device) -> Self {
        BatchNorm2d {
            weight: Tensor::ones(&[1, channels, 1, 1], device).requires_grad_(),
            bias: Tensor::zeros(&[1, channels, 1, 1], device).requires_grad_(),
            running_mean: Tensor::zeros(&[1, channels, 1, 1], device),
            running_var: Tensor::ones(&[1, channels, 1, 1], device),
            momentum: 0.1,
            eps: 1e-5,
            training: AtomicBool::new(true),
        }
    }
}

fn mean_nhw(x: &Tensor) -> Tensor {
    x.mean(0, true).mean(2, true).mean(3, true)
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (mean, var) = if self.training.load(Ordering::Relaxed) {
            let mean = mean_nhw(x);
            let xc = x - &mean;
            let var = mean_nhw(&(&xc * &xc));
            no_grad(|| {
                let n = (x.numel() / x.shape()[1]) as f32;
                let m = self.momentum;
                let unbiased = &var * (n / (n - 1.0).max(1.0));
                self.running_mean.copy_(&(&self.running_mean * (1.0 - m) + &mean * m));
                self.running_var.copy_(&(&self.running_var * (1.0 - m) + unbiased * m));
            });
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };
        let y = (x - &mean) / (var + self.eps).sqrt();
        y * &self.weight + &self.bias
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![("running_mean".into(), self.running_mean.clone()), ("running_var".into(), self.running_var.clone())]
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}

/// Layer normalization over the last dimension.
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, device: Device) -> Self {
        LayerNorm {
            weight: Tensor::ones(&[dim], device).requires_grad_(),
            bias: Tensor::zeros(&[dim], device).requires_grad_(),
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }
}

/// Lookup table of `num` vectors of size `dim`.
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num: usize, dim: usize, device: Device) -> Self {
        Embedding { weight: (Tensor::randn(&[num, dim], device) * 0.02).detach().requires_grad_() }
    }

    /// `ids` laid out as `shape`; output is `shape + [dim]`.
    pub fn forward_ids(&self, ids: &[usize], shape: &[usize]) -> Tensor {
        let mut out_shape = shape.to_vec();
        out_shape.push(self.weight.shape()[1]);
        self.weight.index_select(ids).reshape(&out_shape)
    }
}

impl Module for Embedding {
    /// `x` holds integer ids stored as floats.
    fn forward(&self, x: &Tensor) -> Tensor {
        let ids: Vec<usize> = x.to_vec().iter().map(|&v| v as usize).collect();
        self.forward_ids(&ids, x.shape())
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }
}

pub struct Dropout {
    pub p: f32,
    training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p, training: AtomicBool::new(true) }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::dropout(x, self.p, self.training.load(Ordering::Relaxed))
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}

/// Flatten all dims after the batch dim.
pub struct Flatten;

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.flatten(1)
    }
}

macro_rules! activation {
    ($name:ident, $method:ident) => {
        pub struct $name;
        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                x.$method()
            }
        }
    };
}

activation!(ReLU, relu);
activation!(GELU, gelu);
activation!(Tanh, tanh);
activation!(Sigmoid, sigmoid);

/// Two-layer MLP used inside transformer blocks: Linear -> GELU -> Linear.
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, hidden: usize, device: Device) -> Self {
        FeedForward { fc1: Linear::new(dim, hidden, device), fc2: Linear::new(hidden, dim, device) }
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.fc1.forward(x).gelu())
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("fc1", self.fc1.named_parameters());
        v.extend(prefixed("fc2", self.fc2.named_parameters()));
        v
    }
}
