//! Composite neural-network functions (like `torch.nn.functional`).
//! Everything here is built from differentiable `Tensor` ops, so it works on
//! any backend and gets gradients for free.

use crate::ops::IntoIndices;
use crate::rand;
use crate::tensor::Tensor;

/// Numerically stable softmax along `dim` (fused kernel).
pub fn softmax(x: &Tensor, dim: isize) -> Tensor {
    x.softmax(dim)
}

/// Numerically stable log-softmax along `dim` (fused kernel).
pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    x.log_softmax(dim)
}

/// Cross-entropy between `logits` `[N, C]` and integer class `targets` (N
/// indices: a tensor, `&[usize]`, `&Vec<usize>`, ...), averaged over the batch.
/// `label_smoothing` in [0, 1).
pub fn cross_entropy(logits: &Tensor, targets: impl IntoIndices, label_smoothing: f32) -> Tensor {
    assert_eq!(logits.rank(), 2, "cross_entropy expects [N, C] logits");
    let n = logits.shape()[0];
    let t = targets.into_indices(logits.device());
    assert_eq!(t.numel(), n, "cross_entropy: {} targets for {n} rows", t.numel());
    // losses are always computed in f32 (also under autocast)
    let lsm = logits.to_dtype(crate::DType::F32).log_softmax(-1);
    let nll = -lsm.gather(1, &t.reshape(&[n, 1])).mean_all();
    if label_smoothing > 0.0 {
        let smooth = -lsm.mean(-1, false).mean_all();
        nll * (1.0 - label_smoothing) + smooth * label_smoothing
    } else {
        nll
    }
}

/// Mean squared error.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let d = pred - target;
    (&d * &d).mean_all()
}

/// Binary cross-entropy on logits (stable): mean(softplus(x) - y*x).
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    // softplus(x) = relu(x) + log(1 + exp(-|x|))
    let sp = logits.relu() + (logits.abs().neg().exp() + 1.0).log();
    (sp - logits * target).mean_all()
}

/// 2D convolution. `x`: `[N, C, H, W]`, `weight`: `[OC, C, KH, KW]`, `bias`: `[OC]`.
pub fn conv2d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Tensor {
    assert_eq!(x.rank(), 4, "conv2d expects [N, C, H, W] input, got {:?}", x.shape());
    let ws = weight.shape();
    let (oc, c, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    assert_eq!(x.shape()[1], c, "conv2d: input has {} channels, weight expects {c}", x.shape()[1]);
    let (n, h, w) = (x.shape()[0], x.shape()[2], x.shape()[3]);
    let oh = (h + 2 * padding - kh) / stride + 1;
    let ow = (w + 2 * padding - kw) / stride + 1;
    // under autocast, lower to half before im2col (the patch matrix is the big tensor)
    let x = match crate::tensor::autocast_dtype() {
        Some(d) => x.to_dtype(d),
        None => x.clone(),
    };
    let cols = x.im2col(kh, kw, stride, padding); // [N, C*KH*KW, L]
    let wmat = weight.reshape(&[oc, c * kh * kw]);
    let out = wmat.matmul(&cols).reshape(&[n, oc, oh, ow]); // [N, OC, L] -> NCHW
    match bias {
        Some(b) => {
            let b = b.reshape(&[1, oc, 1, 1]).to_dtype(out.dtype());
            out + b
        }
        None => out,
    }
}

fn pool_patches(x: &Tensor, k: usize, stride: usize, padding: usize) -> (Tensor, [usize; 4]) {
    assert_eq!(x.rank(), 4, "pooling expects [N, C, H, W]");
    let s = x.shape();
    let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
    let oh = (h + 2 * padding - k) / stride + 1;
    let ow = (w + 2 * padding - k) / stride + 1;
    let patches = x.reshape(&[n * c, 1, h, w]).im2col(k, k, stride, padding); // [N*C, k*k, L]
    (patches, [n, c, oh, ow])
}

/// 2D max pooling (square kernel).
pub fn max_pool2d(x: &Tensor, kernel: usize, stride: usize) -> Tensor {
    let (p, out) = pool_patches(x, kernel, stride, 0);
    p.max(1, false).reshape(&out)
}

/// 2D average pooling (square kernel).
pub fn avg_pool2d(x: &Tensor, kernel: usize, stride: usize) -> Tensor {
    let (p, out) = pool_patches(x, kernel, stride, 0);
    p.mean(1, false).reshape(&out)
}

/// Global average pool `[N, C, H, W]` -> `[N, C]`.
pub fn adaptive_avg_pool_1x1(x: &Tensor) -> Tensor {
    let s = x.shape();
    x.reshape(&[s[0], s[1], s[2] * s[3]]).mean(-1, false)
}

/// Layer normalization over the last dimension (fused kernel + affine).
pub fn layer_norm(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Tensor {
    let mut y = x.normalize_last(eps);
    if let Some(w) = weight {
        y = y * w;
    }
    if let Some(b) = bias {
        y = y + b;
    }
    y
}

/// Inverted dropout: zero with probability `p` and scale the rest by 1/(1-p).
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p <= 0.0 {
        return x.clone();
    }
    assert!(p < 1.0, "dropout p must be < 1");
    let keep = 1.0 - p;
    let mask: Vec<f32> = rand::uniform_vec(x.numel(), 0.0, 1.0)
        .into_iter()
        .map(|u| if u < keep { 1.0 / keep } else { 0.0 })
        .collect();
    x * &Tensor::from_vec(mask, x.shape(), x.device()).to_dtype(x.dtype())
}

/// Additive causal mask `[T, T]`: 0 on/below the diagonal, -inf-ish above.
pub fn causal_mask(t: usize, device: crate::Device) -> Tensor {
    let mut m = vec![0.0; t * t];
    for i in 0..t {
        for j in i + 1..t {
            m[i * t + j] = -1e9;
        }
    }
    Tensor::from_vec(m, &[t, t], device)
}

/// Scaled dot-product attention. `q`: `[.., Tq, D]`, `k`, `v`: `[.., Tk, D]`,
/// optional additive `mask` broadcastable to `[.., Tq, Tk]`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    dropout_p: f32,
    training: bool,
) -> Tensor {
    let d = q.dim(-1) as f32;
    let mut scores = q.matmul(&k.t()) * (1.0 / d.sqrt());
    if let Some(m) = mask {
        scores = scores + m;
    }
    let attn = dropout(&softmax(&scores, -1), dropout_p, training);
    attn.matmul(v)
}

/// Fixed sinusoidal positional encoding `[T, D]` (Vaswani et al.).
pub fn sinusoidal_positional_encoding(t: usize, d: usize, device: crate::Device) -> Tensor {
    let mut pe = vec![0.0; t * d];
    for pos in 0..t {
        for i in 0..d {
            let freq = 1.0 / 10000f32.powf((2 * (i / 2)) as f32 / d as f32);
            let a = pos as f32 * freq;
            pe[pos * d + i] = if i % 2 == 0 { a.sin() } else { a.cos() };
        }
    }
    Tensor::from_vec(pe, &[t, d], device)
}

/// Fraction of rows where `argmax(logits) == target` (computed on the device).
pub fn accuracy(logits: &Tensor, targets: impl IntoIndices) -> f32 {
    let pred = logits.argmax(-1, false);
    let t = targets.into_indices(logits.device()).reshape(pred.shape());
    pred.eq_t(&t).mean_all().item()
}
