use super::module::{prefixed, Module};
use crate::device::Device;
use crate::functional as F;
use crate::tensor::{no_grad, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Uniform(-bound, bound) parameter, the PyTorch default for Linear/Conv.
fn uniform_param(shape: &[usize], bound: f32, device: Device) -> Tensor {
    Tensor::rand_uniform(shape, -bound, bound, device).requires_grad_()
}

/// Low-rank adapter (Hu et al. 2021) on a [`Linear`]:
/// `y += (x @ A @ B) * scale`, see [`Linear::add_lora`].
#[derive(Clone)]
pub struct Lora {
    /// `[in, rank]`, random init.
    pub a: Tensor,
    /// `[rank, out]`, zero init (so the adapter starts as a no-op).
    pub b: Tensor,
    /// `alpha / rank`.
    pub scale: f32,
}

/// Fully connected layer `y = x @ W + b`, with `W` stored as `[in, out]`.
/// Cloning shares the parameters (like cloning a [`Tensor`]).
#[derive(Clone)]
pub struct Linear {
    /// `[in, out]` weight matrix.
    pub weight: Tensor,
    /// Optional `[out]` bias.
    pub bias: Option<Tensor>,
    /// Weight-only int8 copy `(q [in, out] I8, scale [out] F32)` after
    /// [`Linear::quantize_int8`]; `weight` is then an empty placeholder.
    int8: Option<(Tensor, Tensor)>,
    /// Optional low-rank adapter.
    pub lora: Option<Lora>,
}

impl Linear {
    /// A layer with bias, initialized like PyTorch (`U(-1/√in, 1/√in)`).
    pub fn new(in_features: usize, out_features: usize, device: Device) -> Self {
        Self::with_bias(in_features, out_features, true, device)
    }

    /// Like [`Linear::new`], with or without a bias.
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool, device: Device) -> Self {
        let bound = 1.0 / (in_features as f32).sqrt();
        Linear {
            weight: uniform_param(&[in_features, out_features], bound, device),
            bias: bias.then(|| uniform_param(&[out_features], bound, device)),
            int8: None,
            lora: None,
        }
    }

    /// Wrap existing tensors (`weight` `[in, out]`).
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Self {
        Linear { weight, bias, int8: None, lora: None }
    }

    /// Whether [`Linear::quantize_int8`] has been applied.
    pub fn is_quantized(&self) -> bool {
        self.int8.is_some()
    }

    /// Attach a trainable LoRA adapter of rank `rank` (`scale = alpha / rank`)
    /// and freeze the layer's own weight and bias, for parameter-efficient
    /// fine-tuning. The output is unchanged until the adapter is trained
    /// (`B` starts at zero). Works on int8-quantized layers too (QLoRA-style).
    /// The adapter is saved as `lora_a` / `lora_b`.
    pub fn add_lora(&mut self, rank: usize, alpha: f32) {
        assert!(rank > 0, "LoRA rank must be positive");
        let (k, n) = match &self.int8 {
            Some((q, _)) => (q.shape()[0], q.shape()[1]),
            None => (self.weight.shape()[0], self.weight.shape()[1]),
        };
        let dev = self.bias.as_ref().map_or_else(|| self.weight.device(), Tensor::device);
        self.weight.set_requires_grad(false);
        if let Some(b) = &self.bias {
            b.set_requires_grad(false);
        }
        self.lora = Some(Lora {
            a: uniform_param(&[k, rank], 1.0 / (k as f32).sqrt(), dev),
            b: Tensor::zeros(&[rank, n], dev).requires_grad_(),
            scale: alpha / rank as f32,
        });
    }

    /// Fold the adapter into the weight (`W += A @ B * scale`) and remove it;
    /// the weight stays frozen. Not possible on an int8 layer.
    pub fn merge_lora(&mut self) {
        let Some(l) = self.lora.take() else { return };
        assert!(self.int8.is_none(), "cannot merge LoRA into an int8 weight");
        let merged = no_grad(|| &self.weight + &(l.a.matmul(&l.b) * l.scale).to_dtype(self.weight.dtype()));
        self.weight.set_storage(merged.storage());
    }

    /// Replace the weight by a symmetric per-output-channel int8 copy
    /// (`scale[j] = max_i |W[i, j]| / 127`) for inference: 4x less weight
    /// memory and traffic. The f32 weight is released; the layer is no longer
    /// trainable. The int8 tensors are saved as buffers `weight_int8` / `weight_scale`.
    pub fn quantize_int8(&mut self) {
        if self.int8.is_some() {
            return;
        }
        let (k, n) = (self.weight.shape()[0], self.weight.shape()[1]);
        let dev = self.weight.device();
        let w = self.weight.to_vec();
        let mut scale = vec![0f32; n];
        for row in w.chunks_exact(n) {
            for (s, &v) in scale.iter_mut().zip(row) {
                *s = s.max(v.abs());
            }
        }
        scale.iter_mut().for_each(|s| *s = if *s > 0.0 { *s / 127.0 } else { 1.0 });
        let q: Vec<i8> = w.iter().enumerate().map(|(i, &v)| (v / scale[i % n]).round().clamp(-127.0, 127.0) as i8).collect();
        self.int8 = Some((Tensor::from_vec_i8(q, &[k, n], dev), Tensor::from_vec(scale, &[n], dev)));
        self.weight = Tensor::zeros(&[0], dev);
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let y = match &self.int8 {
            Some((q, s)) => x.matmul_int8(q, s),
            None => x.matmul(&self.weight), // half under autocast
        };
        let y = match &self.lora {
            Some(l) => {
                let d = x.matmul(&l.a).matmul(&l.b) * l.scale;
                let d = d.to_dtype(y.dtype());
                y + d
            }
            None => y,
        };
        match &self.bias {
            Some(b) => {
                let b = b.to_dtype(y.dtype());
                y + b
            }
            None => y,
        }
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        if self.int8.is_none() {
            v.push(("weight".to_string(), self.weight.clone()));
        }
        if let Some(b) = &self.bias {
            v.push(("bias".to_string(), b.clone()));
        }
        if let Some(l) = &self.lora {
            v.push(("lora_a".to_string(), l.a.clone()));
            v.push(("lora_b".to_string(), l.b.clone()));
        }
        v
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        match &self.int8 {
            Some((q, s)) => vec![("weight_int8".to_string(), q.clone()), ("weight_scale".to_string(), s.clone())],
            None => vec![],
        }
    }
}

/// 2D convolution over `[N, C, H, W]` with a square kernel.
pub struct Conv2d {
    /// `[out_ch, in_ch, k, k]` kernels.
    pub weight: Tensor,
    /// Optional `[out_ch]` bias.
    pub bias: Option<Tensor>,
    /// Step between kernel positions.
    pub stride: usize,
    /// Zero padding on each side.
    pub padding: usize,
}

impl Conv2d {
    /// A convolution with bias (`padding` zeros on each side), PyTorch-style initialization.
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, device: Device) -> Self {
        Self::with_bias(in_ch, out_ch, kernel, stride, padding, true, device)
    }

    /// Like [`Conv2d::new`]; `bias = false` for convolutions followed by batch norm.
    pub fn with_bias(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, bias: bool, device: Device) -> Self {
        let bound = 1.0 / ((in_ch * kernel * kernel) as f32).sqrt();
        Conv2d {
            weight: uniform_param(&[out_ch, in_ch, kernel, kernel], bound, device),
            bias: bias.then(|| uniform_param(&[out_ch], bound, device)),
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

/// 2D max pooling with a square window.
pub struct MaxPool2d {
    /// Window size.
    pub kernel: usize,
    /// Step between windows.
    pub stride: usize,
    /// Implicit `-inf` padding on each side.
    pub padding: usize,
}

impl MaxPool2d {
    /// Max pooling without padding.
    pub fn new(kernel: usize, stride: usize) -> Self {
        MaxPool2d { kernel, stride, padding: 0 }
    }

    /// Max pooling with `padding` (`<= kernel / 2`) on each side.
    pub fn with_padding(kernel: usize, stride: usize, padding: usize) -> Self {
        MaxPool2d { kernel, stride, padding }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::max_pool2d_padded(x, self.kernel, self.stride, self.padding)
    }
}

/// Global average pooling `[N, C, H, W]` -> `[N, C]`.
pub struct GlobalAvgPool;

impl Module for GlobalAvgPool {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::adaptive_avg_pool_1x1(x)
    }
}

/// 2D average pooling with a square window (no padding).
pub struct AvgPool2d {
    /// Window size.
    pub kernel: usize,
    /// Step between windows.
    pub stride: usize,
}

impl AvgPool2d {
    /// Average pooling over `kernel x kernel` windows every `stride` pixels.
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
    /// Learned scale `[1, C, 1, 1]`.
    pub weight: Tensor,
    /// Learned shift `[1, C, 1, 1]`.
    pub bias: Tensor,
    /// Running mean `[1, C, 1, 1]` used in eval mode.
    pub running_mean: Tensor,
    /// Running (unbiased) variance `[1, C, 1, 1]` used in eval mode.
    pub running_var: Tensor,
    /// Weight of the current batch in the running statistics (default 0.1).
    pub momentum: f32,
    /// Added to the variance (default 1e-5).
    pub eps: f32,
    training: AtomicBool,
}

impl BatchNorm2d {
    /// Batch norm over `channels` (scale 1, shift 0).
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
        // batch statistics are computed in f32 (also under autocast)
        let x = &x.to_dtype(self.weight.dtype());
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
#[derive(Clone)]
pub struct LayerNorm {
    /// Learned scale `[dim]`.
    pub weight: Tensor,
    /// Learned shift `[dim]`.
    pub bias: Tensor,
    /// Added to the variance (default 1e-5).
    pub eps: f32,
}

impl LayerNorm {
    /// Layer norm over a last dimension of size `dim` (scale 1, shift 0).
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
#[derive(Clone)]
pub struct Embedding {
    /// `[num, dim]` table.
    pub weight: Tensor,
}

impl Embedding {
    /// A table of `num` vectors of size `dim`, initialized from N(0, 0.02²).
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
    /// `x` holds integer ids (an `I32` tensor, or floats holding integers).
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out_shape = x.shape().to_vec();
        out_shape.push(self.weight.shape()[1]);
        self.weight.index_select(x).reshape(&out_shape)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }
}

/// Inverted dropout (identity in eval mode).
pub struct Dropout {
    /// Probability of zeroing an element.
    pub p: f32,
    training: AtomicBool,
}

impl Dropout {
    /// Dropout with drop probability `p` (training mode).
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
    ($name:ident, $method:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name;
        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                x.$method()
            }
        }
    };
}

activation!(ReLU, relu, "`max(x, 0)` element-wise.");
activation!(GELU, gelu, "GELU (tanh approximation) element-wise.");
activation!(Tanh, tanh, "Hyperbolic tangent element-wise.");
activation!(Sigmoid, sigmoid, "Logistic sigmoid element-wise.");

/// Two-layer MLP used inside transformer blocks: Linear -> GELU -> Linear.
#[derive(Clone)]
pub struct FeedForward {
    /// `[dim, hidden]` input projection.
    pub fc1: Linear,
    /// `[hidden, dim]` output projection.
    pub fc2: Linear,
}

impl FeedForward {
    /// An MLP `dim -> hidden -> dim`.
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

/// RMS normalization over the last dimension with a learned scale (Llama).
#[derive(Clone)]
pub struct RMSNorm {
    /// Learned scale `[dim]`.
    pub weight: Tensor,
    /// Added to the mean square.
    pub eps: f32,
}

impl RMSNorm {
    /// RMS norm over a last dimension of size `dim` (scale 1).
    pub fn new(dim: usize, eps: f32, device: Device) -> Self {
        RMSNorm { weight: Tensor::ones(&[dim], device).requires_grad_(), eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::rms_norm(x, Some(&self.weight), self.eps)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }
}

/// Gated feed-forward (Shazeer 2020) used by Llama:
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`, all without bias.
#[derive(Clone)]
pub struct SwiGLU {
    /// `[dim, hidden]` gate projection (SiLU branch).
    pub gate_proj: Linear,
    /// `[dim, hidden]` linear branch.
    pub up_proj: Linear,
    /// `[hidden, dim]` output projection.
    pub down_proj: Linear,
}

impl SwiGLU {
    /// A gated MLP `dim -> hidden -> dim` without biases.
    pub fn new(dim: usize, hidden: usize, device: Device) -> Self {
        SwiGLU {
            gate_proj: Linear::with_bias(dim, hidden, false, device),
            up_proj: Linear::with_bias(dim, hidden, false, device),
            down_proj: Linear::with_bias(hidden, dim, false, device),
        }
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.gate_proj.forward(x).silu() * self.up_proj.forward(x);
        self.down_proj.forward(&h)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("gate_proj", self.gate_proj.named_parameters());
        v.extend(prefixed("up_proj", self.up_proj.named_parameters()));
        v.extend(prefixed("down_proj", self.down_proj.named_parameters()));
        v
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("gate_proj", self.gate_proj.named_buffers());
        v.extend(prefixed("up_proj", self.up_proj.named_buffers()));
        v.extend(prefixed("down_proj", self.down_proj.named_buffers()));
        v
    }
}
