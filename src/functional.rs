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

/// Cross-entropy like [`cross_entropy`] (no smoothing), averaged only over
/// rows whose target is not `ignore_index` (e.g. padding).
pub fn cross_entropy_ignore(logits: &Tensor, targets: impl IntoIndices, ignore_index: usize) -> Tensor {
    assert_eq!(logits.rank(), 2, "cross_entropy_ignore expects [N, C] logits");
    let n = logits.shape()[0];
    let dev = logits.device();
    let t = targets.into_indices(dev).to_vec_i32();
    assert_eq!(t.len(), n, "cross_entropy_ignore: {} targets for {n} rows", t.len());
    // ignored rows may hold any id: gather class 0 and give it weight 0
    let keep: Vec<f32> = t.iter().map(|&x| if x as usize == ignore_index { 0.0 } else { 1.0 }).collect();
    let safe: Vec<i32> = t.iter().map(|&x| if x as usize == ignore_index { 0 } else { x }).collect();
    let count = keep.iter().sum::<f32>().max(1.0);
    let keep = Tensor::from_vec(keep, &[n, 1], dev);
    let lsm = logits.to_dtype(crate::DType::F32).log_softmax(-1);
    let nll = (lsm.gather(1, &Tensor::from_vec_i32(safe, &[n, 1], dev)) * &keep).sum_all();
    nll * (-1.0 / count)
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

/// 2D max pooling with implicit `-inf` padding of `padding` on each side
/// (like `torch.nn.functional.max_pool2d(..., padding=p)`).
pub fn max_pool2d_padded(x: &Tensor, kernel: usize, stride: usize, padding: usize) -> Tensor {
    if padding == 0 {
        return max_pool2d(x, kernel, stride);
    }
    assert!(padding <= kernel / 2, "max_pool2d: padding {padding} must be at most half the kernel {kernel}");
    let (p, out) = pool_patches(x, kernel, stride, padding);
    // im2col pads with zeros: mask those positions out with a very negative value
    let valid = crate::no_grad(|| pool_patches(&Tensor::ones(x.shape(), x.device()), kernel, stride, padding).0);
    p.masked_fill(&valid.lt_scalar(0.5), -1e30).max(1, false).reshape(&out)
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

/// RMS normalization over the last dimension (Zhang & Sennrich, used by
/// Llama): `x / sqrt(mean(x²) + eps) * weight`. The statistics are computed
/// in f32; the result has the input's dtype (f32 under autocast).
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Tensor {
    let xf = x.to_dtype(crate::DType::F32);
    let rstd = (&xf * &xf).mean(-1, true).add_scalar(eps).sqrt().recip();
    let mut y = &xf * &rstd;
    if crate::tensor::autocast_dtype().is_none() {
        y = y.to_dtype(x.dtype());
    }
    match weight {
        Some(w) => {
            let w = w.to_dtype(y.dtype());
            y * w
        }
        None => y,
    }
}

/// Rotary-embedding inverse frequencies `theta^(-2i/head_dim)`, `i < head_dim/2`.
pub fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    assert!(head_dim.is_multiple_of(2), "RoPE needs an even head_dim, got {head_dim}");
    (0..head_dim / 2).map(|i| theta.powf(-((2 * i) as f32) / head_dim as f32)).collect()
}

/// RoPE tables `(cos, sin)`, each `[T, 2 * inv_freq.len()]`, for positions
/// `start..start + t` (the frequencies are repeated for both halves, as in
/// the Hugging Face "rotate_half" layout).
pub fn rope_cos_sin(inv_freq: &[f32], start: usize, t: usize, device: crate::Device) -> (Tensor, Tensor) {
    let half = inv_freq.len();
    let d = 2 * half;
    let (mut cos, mut sin) = (vec![0.0; t * d], vec![0.0; t * d]);
    for p in 0..t {
        for (i, &f) in inv_freq.iter().enumerate() {
            // f64 keeps large positions accurate
            let a = (start + p) as f64 * f as f64;
            let (s, c) = a.sin_cos();
            for j in [i, i + half] {
                cos[p * d + j] = c as f32;
                sin[p * d + j] = s as f32;
            }
        }
    }
    (Tensor::from_vec(cos, &[t, d], device), Tensor::from_vec(sin, &[t, d], device))
}

/// Apply rotary position embeddings to `x` `[.., T, D]` with tables from
/// [`rope_cos_sin`]: `x * cos + rotate_half(x) * sin`, where
/// `rotate_half([x1, x2]) = [-x2, x1]`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let d = x.dim(-1);
    assert_eq!(cos.dim(-1), d, "apply_rope: tables have dim {}, input {d}", cos.dim(-1));
    let half = d / 2;
    let rot = Tensor::cat(&[&x.narrow(-1, half, half).neg(), &x.narrow(-1, 0, half)], -1);
    let (cos, sin) = (cos.to_dtype(x.dtype()), sin.to_dtype(x.dtype()));
    x * &cos + rot * sin
}

/// Repeat key/value heads for grouped-query attention:
/// `[B, H_kv, T, D]` -> `[B, H_kv * n_rep, T, D]` (each head `n_rep` times in a row).
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return x.clone();
    }
    let s = x.shape();
    let (b, h, t, d) = (s[0], s[1], s[2], s[3]);
    x.unsqueeze(2).expand(&[b, h, n_rep, t, d]).reshape(&[b, h * n_rep, t, d])
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
    causal_mask_offset(t, 0, device)
}

/// Causal mask `[Tq, past + Tq]` for `Tq` new queries attending to `past`
/// cached positions plus themselves: query `i` sees keys `0..=past + i`.
pub fn causal_mask_offset(tq: usize, past: usize, device: crate::Device) -> Tensor {
    let tk = past + tq;
    let mut m = vec![0.0; tq * tk];
    for i in 0..tq {
        for j in past + i + 1..tk {
            m[i * tk + j] = -1e9;
        }
    }
    Tensor::from_vec(m, &[tq, tk], device)
}

/// Additive key-padding mask `[B, 1, 1, T]` for `tokens` `[B, T]`: `-1e9`
/// where the token is `pad_id`, `0` elsewhere (broadcasts over heads and queries).
pub fn padding_mask(tokens: &Tensor, pad_id: usize) -> Tensor {
    let (b, t) = (tokens.shape()[0], tokens.shape()[1]);
    let m: Vec<f32> = tokens.to_vec_i32().iter().map(|&x| if x as usize == pad_id { -1e9 } else { 0.0 }).collect();
    Tensor::from_vec(m, &[b, 1, 1, t], tokens.device())
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
