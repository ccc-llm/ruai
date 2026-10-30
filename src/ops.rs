//! Differentiable tensor operations. Each op computes its forward result with
//! backend primitives and registers a backward closure built from other ops.

use crate::backend::{BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_shape, norm_dim, numel};
use crate::tensor::{autocast_dtype, Tensor};

fn need(t: &Tensor) -> bool {
    t.requires_grad()
}

/// Inside a backward function: whether gradients are being built with their
/// own graph (`create_graph`), so they must use differentiable ops instead of
/// raw kernels.
fn graph() -> bool {
    crate::tensor::is_grad_enabled()
}

fn same_device(a: &Tensor, b: &Tensor) {
    assert_eq!(a.device(), b.device(), "tensors are on different devices ({} vs {})", a.device(), b.device());
}

fn assert_float(t: &Tensor, op: &str) {
    assert!(t.dtype().is_float(), "{op}: expected a floating-point tensor, got {}", t.dtype());
}

/// Cast both operands to their promoted dtype ([`DType::promote`]).
fn promoted(a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    if a.dtype() == b.dtype() {
        return (a.clone(), b.clone());
    }
    let d = DType::promote(a.dtype(), a.rank() == 0, b.dtype(), b.rank() == 0);
    (a.to_dtype(d), b.to_dtype(d))
}

/// Anything usable as an index tensor: `&Tensor` (any dtype), `&[usize]`,
/// `&Vec<usize>`, `&[usize; N]`. Converted to an `I32` tensor on `device`.
pub trait IntoIndices {
    /// Convert to an `I32` index tensor on `device`.
    fn into_indices(self, device: Device) -> Tensor;
}

impl IntoIndices for &Tensor {
    fn into_indices(self, device: Device) -> Tensor {
        self.to_dtype(DType::I32).to(device)
    }
}

impl IntoIndices for &[usize] {
    fn into_indices(self, device: Device) -> Tensor {
        Tensor::from_indices(self, &[self.len()], device)
    }
}

impl IntoIndices for &Vec<usize> {
    fn into_indices(self, device: Device) -> Tensor {
        self.as_slice().into_indices(device)
    }
}

impl<const N: usize> IntoIndices for &[usize; N] {
    fn into_indices(self, device: Device) -> Tensor {
        self.as_slice().into_indices(device)
    }
}

impl Tensor {
    // ------------------------------------------------------------------
    // elementwise unary
    // ------------------------------------------------------------------

    fn unary_raw(&self, op: UnaryOp) -> Tensor {
        assert_float(self, "unary op");
        Tensor::from_storage(self.backend().unary(op, &self.storage()), self.shape().to_vec())
    }

    /// `backward(g, x, y)` gets the output `y`: a detached copy normally, or
    /// `fwd(x)` recomputed with its graph when building a differentiable gradient.
    fn unary_op<F>(&self, op: UnaryOp, name: &'static str, fwd: fn(&Tensor) -> Tensor, backward: F) -> Tensor
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Send + Sync + 'static,
    {
        let out = self.unary_raw(op);
        let x = self.clone();
        let y = out.clone(); // detached copy of the output for the backward pass
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![self.clone()], name, move |g| {
            let y = if graph() { fwd(&x) } else { y.clone() };
            vec![Some(backward(g, &x, &y))]
        })
    }

    /// Elementwise `-x`.
    pub fn neg(&self) -> Tensor {
        self.unary_op(UnaryOp::Neg, "neg", Tensor::neg, |g, _, _| g.neg())
    }

    /// Elementwise `e^x`.
    pub fn exp(&self) -> Tensor {
        self.unary_op(UnaryOp::Exp, "exp", Tensor::exp, |g, _, y| g * y)
    }

    /// Elementwise natural logarithm.
    pub fn log(&self) -> Tensor {
        self.unary_op(UnaryOp::Log, "log", Tensor::log, |g, x, _| g / x)
    }

    /// Elementwise square root.
    pub fn sqrt(&self) -> Tensor {
        self.unary_op(UnaryOp::Sqrt, "sqrt", Tensor::sqrt, |g, _, y| g * &(y.recip() * 0.5))
    }

    /// Elementwise `1 / x`.
    pub fn recip(&self) -> Tensor {
        self.unary_op(UnaryOp::Recip, "recip", Tensor::recip, |g, _, y| (g * &(y * y)).neg())
    }

    /// Elementwise `|x|` (gradient `sign(x)`, 0 at 0).
    pub fn abs(&self) -> Tensor {
        self.unary_op(UnaryOp::Abs, "abs", Tensor::abs, |g, x, _| {
            let sign = &x.gt_scalar(0.0) - &x.lt_scalar(0.0);
            g * &sign
        })
    }

    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        self.unary_op(UnaryOp::Tanh, "tanh", Tensor::tanh, |g, _, y| g * &((y * y).neg() + 1.0))
    }

    /// Elementwise logistic sigmoid `1 / (1 + e^-x)`.
    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(UnaryOp::Sigmoid, "sigmoid", Tensor::sigmoid, |g, _, y| g * &(y * &(y.neg() + 1.0)))
    }

    /// Elementwise `max(x, 0)`.
    pub fn relu(&self) -> Tensor {
        self.unary_op(UnaryOp::Relu, "relu", Tensor::relu, |g, x, _| g * &x.gt_scalar(0.0))
    }

    /// GELU (tanh approximation), as used in GPT/BERT.
    pub fn gelu(&self) -> Tensor {
        self.unary_op(UnaryOp::Gelu, "gelu", Tensor::gelu, |g, x, _| {
            if !graph() {
                return g * &x.unary_raw(UnaryOp::GeluGrad);
            }
            // d/dx of 0.5 x (1 + tanh(u)), u = c (x + 0.044715 x³), from differentiable ops
            const C: f32 = 0.797_884_6; // sqrt(2 / pi)
            let x2 = x * x;
            let t = ((x + &(&x2 * x) * 0.044715) * C).tanh();
            let du = (x2 * (3.0 * 0.044715) + 1.0) * C;
            let d = (&t + 1.0) * 0.5 + (x * &((&t * &t).neg() + 1.0)) * &du * 0.5;
            g * &d
        })
    }

    /// SiLU / swish: `x * sigmoid(x)` (composed, so it works on every backend).
    pub fn silu(&self) -> Tensor {
        self * &self.sigmoid()
    }

    /// Elementwise sine.
    pub fn sin(&self) -> Tensor {
        self.unary_op(UnaryOp::Sin, "sin", Tensor::sin, |g, x, _| g * &x.cos())
    }

    /// Elementwise cosine.
    pub fn cos(&self) -> Tensor {
        self.unary_op(UnaryOp::Cos, "cos", Tensor::cos, |g, x, _| (g * &x.sin()).neg())
    }

    // ------------------------------------------------------------------
    // elementwise binary (with broadcasting)
    // ------------------------------------------------------------------

    fn binary_raw(&self, op: BinaryOp, other: &Tensor) -> Tensor {
        same_device(self, other);
        let (a, b) = promoted(self, other);
        let (a, b) = if a.dtype() == DType::I32 && !matches!(op, BinaryOp::Eq | BinaryOp::Gt | BinaryOp::Lt) {
            (a.to_dtype(DType::F32), b.to_dtype(DType::F32))
        } else {
            (a, b)
        };
        a.binary_raw_same(op, &b)
    }

    fn binary_raw_same(&self, op: BinaryOp, other: &Tensor) -> Tensor {
        let out_shape = broadcast_shape(self.shape(), other.shape());
        let s = self.backend().binary(op, &self.storage(), self.shape(), &other.storage(), other.shape(), &out_shape);
        Tensor::from_storage(s, out_shape)
    }

    fn binary_op<F>(&self, op: BinaryOp, other: &Tensor, name: &'static str, backward: F) -> Tensor
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> (Option<Tensor>, Option<Tensor>) + Send + Sync + 'static,
    {
        same_device(self, other);
        let (a, b) = promoted(self, other);
        let out = a.binary_raw(op, &b);
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![a.clone(), b.clone()], name, move |g| {
            let (ga, gb) = backward(g, &a, &b);
            vec![ga.map(|t| t.sum_to(a.shape())), gb.map(|t| t.sum_to(b.shape()))]
        })
    }

    /// `self + other` with broadcasting and dtype promotion.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Add, other, "add", |g, a, b| {
            (need(a).then(|| g.clone()), need(b).then(|| g.clone()))
        })
    }

    /// `self - other` with broadcasting and dtype promotion.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Sub, other, "sub", |g, a, b| (need(a).then(|| g.clone()), need(b).then(|| g.neg())))
    }

    /// `self * other` with broadcasting and dtype promotion.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Mul, other, "mul", |g, a, b| (need(a).then(|| g * b), need(b).then(|| g * a)))
    }

    /// `self / other` with broadcasting and dtype promotion.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Div, other, "div", |g, a, b| {
            (need(a).then(|| g / b), need(b).then(|| (&(g * a) / &(b * b)).neg()))
        })
    }

    /// Elementwise maximum; the gradient goes to the larger input.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Maximum, other, "maximum", |g, a, b| {
            let mask = a.binary_raw(BinaryOp::Lt, b); // 1 where b wins
            (need(a).then(|| g * &(mask.neg() + 1.0)), need(b).then(|| g * &mask))
        })
    }

    /// Elementwise minimum; the gradient goes to the smaller input.
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Minimum, other, "minimum", |g, a, b| {
            let mask = a.binary_raw(BinaryOp::Gt, b); // 1 where b wins
            (need(a).then(|| g * &(mask.neg() + 1.0)), need(b).then(|| g * &mask))
        })
    }

    /// Clamp into `[min, max]`.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        self.maximum(&self.scalar_like(min)).minimum(&self.scalar_like(max))
    }

    /// `cond != 0 ? a : b` with broadcasting (differentiable in `a` and `b`).
    pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        same_device(cond, a);
        same_device(a, b);
        let shape = broadcast_shape(&broadcast_shape(cond.shape(), a.shape()), b.shape());
        let (a, b) = promoted(a, b);
        let c = cond.detach().to_dtype(a.dtype()).expand(&shape);
        let (ae, bexp) = (a.expand(&shape), b.expand(&shape));
        let s = a.backend().where_cond(&c.storage(), &ae.storage(), &bexp.storage());
        Tensor::from_op(s, shape, vec![ae, bexp], "where", move |g| {
            let z = Tensor::zeros_like(g);
            if graph() {
                return vec![Some(Tensor::where_cond(&c, g, &z)), Some(Tensor::where_cond(&c, &z, g))];
            }
            let pick = |x: &Tensor, y: &Tensor| {
                let s = g.backend().where_cond(&c.storage(), &x.storage(), &y.storage());
                Tensor::from_storage(s, g.shape().to_vec())
            };
            vec![Some(pick(g, &z)), Some(pick(&z, g))]
        })
    }

    /// Replace elements where `mask != 0` with `value`.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        Tensor::where_cond(mask, &self.scalar_like(value), self)
    }

    /// `self ^ p` for a constant exponent.
    pub fn powf(&self, p: f32) -> Tensor {
        let e = Tensor::scalar(p, self.device());
        let out = self.binary_raw(BinaryOp::Pow, &e);
        let x = self.clone();
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![self.clone()], "powf", move |g| {
            let d = if graph() { x.powf(p - 1.0) } else { x.binary_raw(BinaryOp::Pow, &Tensor::scalar(p - 1.0, x.device())) } * p;
            vec![Some(g * &d)]
        })
    }

    fn scalar_like(&self, v: f32) -> Tensor {
        Tensor::scalar(v, self.device())
    }

    /// `self + v`.
    pub fn add_scalar(&self, v: f32) -> Tensor {
        self.add(&self.scalar_like(v))
    }

    /// `self * v`.
    pub fn mul_scalar(&self, v: f32) -> Tensor {
        self.mul(&self.scalar_like(v))
    }

    // comparisons (not differentiable; return 0/1 tensors)
    /// 1 where `self == other`, else 0 (broadcasting; not differentiable).
    pub fn eq_t(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Eq, other)
    }

    /// 1 where `self > other`, else 0 (broadcasting; not differentiable).
    pub fn gt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Gt, other)
    }

    /// 1 where `self < other`, else 0 (broadcasting; not differentiable).
    pub fn lt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Lt, other)
    }

    /// 1 where `self > v`, else 0 (not differentiable).
    pub fn gt_scalar(&self, v: f32) -> Tensor {
        self.gt(&self.scalar_like(v))
    }

    /// 1 where `self < v`, else 0 (not differentiable).
    pub fn lt_scalar(&self, v: f32) -> Tensor {
        self.lt(&self.scalar_like(v))
    }

    // ------------------------------------------------------------------
    // broadcasting helpers
    // ------------------------------------------------------------------

    /// Broadcast to `shape` (a view: broadcast dims get stride 0, no copy).
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        if self.shape() == shape {
            return self.clone();
        }
        assert_eq!(broadcast_shape(self.shape(), shape), shape, "cannot expand {:?} to {shape:?}", self.shape());
        let (_, st, off) = self.strided_parts();
        let lead = shape.len() - self.rank();
        let strides: Vec<usize> = (0..shape.len())
            .map(|d| if d < lead || self.shape()[d - lead] != shape[d] { 0 } else { st[d - lead] })
            .collect();
        let in_shape = self.shape().to_vec();
        Tensor::view_op(self, shape.to_vec(), strides, off, "expand", move |g| vec![Some(g.sum_to(&in_shape))])
    }

    /// Sum a broadcast result back down to `shape` (inverse of broadcasting).
    pub fn sum_to(&self, shape: &[usize]) -> Tensor {
        if self.shape() == shape {
            return self.clone();
        }
        let mut t = self.clone();
        while t.rank() > shape.len() {
            t = t.sum(0, false);
        }
        for (i, &d) in shape.iter().enumerate() {
            if d == 1 && t.shape()[i] != 1 {
                t = t.sum(i as isize, true);
            }
        }
        assert_eq!(t.shape(), shape, "sum_to: cannot reduce {:?} to {shape:?}", self.shape());
        t
    }

    // ------------------------------------------------------------------
    // reductions
    // ------------------------------------------------------------------

    fn reduced_shapes(&self, dim: isize) -> (usize, Vec<usize>, Vec<usize>) {
        let ax = norm_dim(dim, self.rank());
        let mut keep = self.shape().to_vec();
        keep[ax] = 1;
        let mut drop = self.shape().to_vec();
        drop.remove(ax);
        (ax, keep, drop)
    }

    /// Sum along `dim` (negative counts from the end); `keepdim` keeps it as size 1.
    pub fn sum(&self, dim: isize, keepdim: bool) -> Tensor {
        let (ax, keep, drop) = self.reduced_shapes(dim);
        let s = self.backend().reduce(ReduceOp::Sum, &self.storage(), self.shape(), ax);
        let in_shape = self.shape().to_vec();
        let keep2 = keep.clone();
        let out_shape = if keepdim { keep } else { drop };
        Tensor::from_op(s, out_shape, vec![self.clone()], "sum", move |g| {
            vec![Some(g.reshape(&keep2).expand(&in_shape))]
        })
    }

    /// Sum of all elements (rank-0 result).
    pub fn sum_all(&self) -> Tensor {
        self.reshape(&[self.numel()]).sum(0, false)
    }

    /// Mean along `dim`.
    pub fn mean(&self, dim: isize, keepdim: bool) -> Tensor {
        let n = self.dim(dim) as f32;
        self.sum(dim, keepdim) * (1.0 / n)
    }

    /// Mean of all elements (rank-0 result).
    pub fn mean_all(&self) -> Tensor {
        self.sum_all() * (1.0 / self.numel() as f32)
    }

    /// Biased variance along `dim`.
    pub fn var(&self, dim: isize, keepdim: bool) -> Tensor {
        let mu = self.mean(dim, true);
        let d = self - &mu;
        (&d * &d).mean(dim, keepdim)
    }

    /// Max along `dim`; the gradient flows to the maximal element(s).
    pub fn max(&self, dim: isize, keepdim: bool) -> Tensor {
        let (ax, keep, drop) = self.reduced_shapes(dim);
        let s = self.backend().reduce(ReduceOp::Max, &self.storage(), self.shape(), ax);
        let x = self.clone();
        let out_keep = Tensor::from_storage(s.clone(), keep.clone());
        let out_shape = if keepdim { keep.clone() } else { drop };
        Tensor::from_op(s, out_shape, vec![self.clone()], "max", move |g| {
            let mask = x.eq_t(&out_keep);
            vec![Some(&g.reshape(&keep) * &mask)]
        })
    }

    /// Min along `dim`; the gradient flows to the minimal element(s).
    pub fn min(&self, dim: isize, keepdim: bool) -> Tensor {
        self.neg().max(dim, keepdim).neg()
    }

    /// Index of the max along `dim` as an `I32` tensor (not differentiable).
    pub fn argmax(&self, dim: isize, keepdim: bool) -> Tensor {
        let (ax, keep, drop) = self.reduced_shapes(dim);
        let s = self.backend().reduce(ReduceOp::ArgMax, &self.storage(), self.shape(), ax);
        Tensor::from_storage(s, if keepdim { keep } else { drop })
    }

    // ------------------------------------------------------------------
    // shape manipulation
    // ------------------------------------------------------------------

    /// Reshape: a view (no copy) when the elements are in row-major order,
    /// otherwise a contiguous copy.
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(numel(shape), self.numel(), "cannot reshape {:?} to {shape:?}", self.shape());
        if shape == self.shape() {
            return self.clone();
        }
        let in_shape = self.shape().to_vec();
        let back = move |g: &Tensor| vec![Some(g.reshape(&in_shape))];
        let (_, st, off) = self.strided_parts();
        if crate::tensor::is_row_major(self.shape(), &st) {
            Tensor::view_op(self, shape.to_vec(), crate::tensor::contiguous_strides(shape), off, "reshape", back)
        } else {
            Tensor::from_op(self.storage(), shape.to_vec(), vec![self.clone()], "reshape", back)
        }
    }

    /// A contiguous (row-major, own buffer) version of this tensor; itself if
    /// already contiguous.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        Tensor::from_op(self.storage(), self.shape().to_vec(), vec![self.clone()], "contiguous", |g| vec![Some(g.clone())])
    }

    /// Reshape where one dimension may be `-1` (inferred), like `torch.view`.
    pub fn view(&self, shape: &[isize]) -> Tensor {
        let known: usize = shape.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
        let resolved: Vec<usize> = shape
            .iter()
            .map(|&d| if d < 0 { self.numel() / known.max(1) } else { d as usize })
            .collect();
        self.reshape(&resolved)
    }

    /// Merge dims `start_dim..` into one: `[B, C, H, W]` -> `[B, C*H*W]` for `start_dim = 1`.
    pub fn flatten(&self, start_dim: usize) -> Tensor {
        let mut s = self.shape()[..start_dim].to_vec();
        s.push(self.shape()[start_dim..].iter().product());
        self.reshape(&s)
    }

    /// Insert a size-1 dim at `dim` (a view).
    pub fn unsqueeze(&self, dim: isize) -> Tensor {
        let d = norm_dim(dim, self.rank() + 1);
        let (_, mut st, off) = self.strided_parts();
        let mut s = self.shape().to_vec();
        s.insert(d, 1);
        st.insert(d, 0);
        Tensor::view_op(self, s, st, off, "unsqueeze", move |g| vec![Some(g.squeeze(d as isize))])
    }

    /// Remove dim `dim`, which must have size 1 (panics otherwise; a view).
    pub fn squeeze(&self, dim: isize) -> Tensor {
        let d = norm_dim(dim, self.rank());
        assert_eq!(self.shape()[d], 1, "squeeze: dim {d} has size {}", self.shape()[d]);
        let (_, mut st, off) = self.strided_parts();
        let mut s = self.shape().to_vec();
        s.remove(d);
        st.remove(d);
        Tensor::view_op(self, s, st, off, "squeeze", move |g| vec![Some(g.unsqueeze(d as isize))])
    }

    /// Reorder dims: output dim `i` is input dim `perm[i]` (a view: only the
    /// strides are permuted).
    pub fn permute(&self, perm: &[usize]) -> Tensor {
        assert_eq!(perm.len(), self.rank(), "permute: wrong number of dims");
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return self.clone();
        }
        let (_, st, off) = self.strided_parts();
        let out_shape: Vec<usize> = perm.iter().map(|&p| self.shape()[p]).collect();
        let strides: Vec<usize> = perm.iter().map(|&p| st[p]).collect();
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        Tensor::view_op(self, out_shape, strides, off, "permute", move |g| vec![Some(g.permute(&inv))])
    }

    /// Swap dims `d0` and `d1` (a view).
    pub fn transpose(&self, d0: isize, d1: isize) -> Tensor {
        let (a, b) = (norm_dim(d0, self.rank()), norm_dim(d1, self.rank()));
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }

    /// Transpose of the last two dims.
    pub fn t(&self) -> Tensor {
        self.transpose(-2, -1)
    }

    // ------------------------------------------------------------------
    // matmul
    // ------------------------------------------------------------------

    /// How this tensor can feed a GEMM without copying: `(buffer, transposed,
    /// batch stride)` when its offset is 0, its last two dims are row-major
    /// (or column-major = `transposed`) and its leading dims collapse to one
    /// stride (0 = broadcast). `None` means it must be materialized.
    pub(crate) fn gemm_operand(&self) -> Option<(crate::backend::Storage, bool, usize)> {
        let (s, st, off) = self.strided_parts();
        let r = self.rank();
        if off != 0 || r < 2 {
            return None;
        }
        let sh = self.shape();
        let (rows, cols, sr, sc) = (sh[r - 2], sh[r - 1], st[r - 2], st[r - 1]);
        let transposed = if (cols == 1 || sc == 1) && (rows == 1 || sr == cols) {
            false
        } else if (rows == 1 || sr == 1) && (cols == 1 || sc == rows) {
            true
        } else {
            return None;
        };
        let mut batch_stride: Option<usize> = None;
        let mut expect = 0;
        for d in (0..r - 2).rev() {
            if sh[d] == 1 {
                continue;
            }
            match batch_stride {
                None => batch_stride = Some(st[d]),
                Some(_) if st[d] != expect => return None,
                Some(_) => {}
            }
            expect = st[d] * sh[d];
        }
        Some((s, transposed, batch_stride.unwrap_or(0)))
    }

    /// Batched matmul without autograd, with optional transposes of the last two dims.
    /// Supports: 2D @ 2D, ND @ 2D, 2D @ ND, and ND @ ND with equal batch dims.
    /// Transposed and batch-strided views (e.g. `k.t()`, a narrowed KV cache)
    /// are read in place.
    pub(crate) fn matmul_raw(&self, other: &Tensor, ta: bool, tb: bool) -> Tensor {
        same_device(self, other);
        let (ash, bsh) = (self.shape(), other.shape());
        assert!(ash.len() >= 2 && bsh.len() >= 2, "matmul needs rank >= 2, got {ash:?} @ {bsh:?}");
        let (ar, ac) = (ash[ash.len() - 2], ash[ash.len() - 1]);
        let (br, bc) = (bsh[bsh.len() - 2], bsh[bsh.len() - 1]);
        let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
        let (k2, n) = if tb { (bc, br) } else { (br, bc) };
        assert_eq!(k, k2, "matmul inner dims differ: {ash:?} @ {bsh:?} (ta={ta}, tb={tb})");
        let abatch = &ash[..ash.len() - 2];
        let bbatch = &bsh[..bsh.len() - 2];
        // operands as stored: (buffer, stored transposed, batch stride)
        let operand = |t: &Tensor, rows: usize, cols: usize| t.gemm_operand().unwrap_or_else(|| (t.storage(), false, rows * cols));
        let (sa, a_t, a_bs) = operand(self, ar, ac);
        let (sb, b_t, b_bs) = operand(other, br, bc);
        let (ta, tb) = (ta ^ a_t, tb ^ b_t);

        let (spec, batch_shape) = if bbatch.is_empty() && !ta && !abatch.is_empty() && a_bs == m * k {
            // A's batch dims are contiguous rows: one big GEMM
            let rows = numel(abatch) * m;
            let spec = MatmulSpec { batch: 1, m: rows, k, n, trans_a: false, trans_b: tb, a_batch_stride: 0, b_batch_stride: 0 };
            (spec, abatch.to_vec())
        } else if abatch == bbatch {
            let b = numel(abatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: a_bs, b_batch_stride: b_bs };
            (spec, abatch.to_vec())
        } else if abatch.is_empty() {
            let b = numel(bbatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: 0, b_batch_stride: b_bs };
            (spec, bbatch.to_vec())
        } else if bbatch.is_empty() {
            let b = numel(abatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: a_bs, b_batch_stride: 0 };
            (spec, abatch.to_vec())
        } else {
            panic!("matmul: unsupported batch broadcast {ash:?} @ {bsh:?}");
        };
        let s = self.backend().matmul(&sa, &sb, spec);
        let mut out_shape = batch_shape;
        out_shape.extend([m, n]);
        Tensor::from_storage(s, out_shape)
    }

    /// Operands of a matmul after autocast / dtype promotion.
    fn matmul_operands(&self, other: &Tensor) -> (Tensor, Tensor) {
        match autocast_dtype() {
            Some(d) if self.dtype().is_float() && other.dtype().is_float() => (self.to_dtype(d), other.to_dtype(d)),
            _ => promoted(self, other),
        }
    }

    /// `self @ otherᵀ` (transpose of the last two dims of `other`) without
    /// materializing the transpose, e.g. a tied output head
    /// `hidden.matmul_nt(&embedding_table)`.
    pub fn matmul_nt(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.matmul_operands(other);
        let out = a.matmul_raw(&b, false, true);
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![a.clone(), b.clone()], "matmul_nt", move |g| {
            // out = a bᵀ:  da = g b,  db = gᵀ a
            let mm = |x: &Tensor, y: &Tensor, tx: bool| {
                if graph() {
                    // differentiable, for second derivatives
                    if tx { x.t().matmul(y) } else { x.matmul(y) }
                } else {
                    x.matmul_raw(y, tx, false)
                }
            };
            let ga = need(&a).then(|| mm(g, &b, false).sum_to(a.shape()));
            let gb = need(&b).then(|| {
                if b.rank() == 2 && a.rank() > 2 {
                    let k = a.dim(-1);
                    let n = g.dim(-1);
                    let af = a.reshape(&[a.numel() / k, k]);
                    let gf = g.reshape(&[g.numel() / n, n]);
                    mm(&gf, &af, true)
                } else {
                    mm(g, &a, true).sum_to(b.shape())
                }
            });
            vec![ga, gb]
        })
    }

    /// Matrix product of the last two dims, batched over leading dims.
    /// Inside [`crate::autocast`] both operands are cast to the autocast dtype.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.matmul_operands(other);
        let out = a.matmul_raw(&b, false, false);
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![a.clone(), b.clone()], "matmul", move |g| {
            // out = a b:  da = g bᵀ,  db = aᵀ g
            let g2 = graph();
            let ga = need(&a).then(|| if g2 { g.matmul_nt(&b) } else { g.matmul_raw(&b, false, true) }.sum_to(a.shape()));
            let gb = need(&b).then(|| {
                if b.rank() == 2 && a.rank() > 2 {
                    let k = a.dim(-1);
                    let n = g.dim(-1);
                    let af = a.reshape(&[a.numel() / k, k]);
                    let gf = g.reshape(&[g.numel() / n, n]);
                    if g2 { af.t().matmul(&gf) } else { af.matmul_raw(&gf, true, false) }
                } else if g2 {
                    a.t().matmul(g).sum_to(b.shape())
                } else {
                    a.matmul_raw(g, true, false).sum_to(b.shape())
                }
            });
            vec![ga, gb]
        })
    }

    /// Weight-only int8 matmul `self @ (q * scales)` for inference:
    /// `self` `[..., k]` (float), `q` `[k, n]` (`I8`), `scales` `[n]` (`F32`).
    /// Differentiable with respect to `self` only (the quantized weights are
    /// frozen), so LoRA adapters can be trained on top of an int8 model.
    pub fn matmul_int8(&self, q: &Tensor, scales: &Tensor) -> Tensor {
        assert_float(self, "matmul_int8");
        assert_eq!(q.dtype(), DType::I8, "matmul_int8: weights must be I8");
        assert_eq!(q.rank(), 2, "matmul_int8: weights must be [k, n]");
        let (k, n) = (q.shape()[0], q.shape()[1]);
        assert_eq!(self.dim(-1), k, "matmul_int8: inner dims differ ({:?} @ {:?})", self.shape(), q.shape());
        assert_eq!(scales.shape(), &[n], "matmul_int8: scales must be [n]");
        let m = self.numel() / k.max(1);
        let s = self.backend().matmul_int8(&self.storage(), &q.storage(), &scales.to_dtype(DType::F32).storage(), m, k, n);
        let mut shape = self.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        let (q, scales) = (q.clone(), scales.clone());
        Tensor::from_op(s, shape, vec![self.clone()], "matmul_int8", move |g| {
            // dx = g @ Wᵀ with the dequantized W = q * scales (materialized only here)
            let w = q.to_dtype(DType::F32) * scales.to_dtype(DType::F32);
            vec![Some(g.matmul_nt(&w))]
        })
    }

    /// Group-wise int4 weight-only matmul `self @ dequant(q, scales)`:
    /// `self` `[..., k]` (float), `q` `[k / 2, n]` (`I8`, two 4-bit values
    /// per byte, see [`crate::backend::Backend::dequant_int4`]), `scales`
    /// `[k / group, n]` (`F32`). Differentiable with respect to `self` only.
    pub fn matmul_int4(&self, q: &Tensor, scales: &Tensor) -> Tensor {
        assert_float(self, "matmul_int4");
        assert_eq!(q.dtype(), DType::I8, "matmul_int4: packed weights must be I8");
        assert!(q.rank() == 2 && scales.rank() == 2, "matmul_int4: q must be [k/2, n] and scales [k/group, n]");
        let (k, n) = (2 * q.shape()[0], q.shape()[1]);
        assert_eq!(self.dim(-1), k, "matmul_int4: inner dims differ ({:?} @ int4 [{k}, {n}])", self.shape());
        assert_eq!(scales.dim(1), n, "matmul_int4: scales must have {n} columns");
        let group = k / scales.dim(0);
        assert_eq!(group * scales.dim(0), k, "matmul_int4: {} scale rows do not divide k = {k}", scales.dim(0));
        let m = self.numel() / k.max(1);
        let sc = scales.to_dtype(DType::F32);
        let s = self.backend().matmul_int4(&self.storage(), &q.storage(), &sc.storage(), m, k, n, group);
        let mut shape = self.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        let q = q.clone();
        Tensor::from_op(s, shape, vec![self.clone()], "matmul_int4", move |g| {
            let w = Tensor::from_storage(g.backend().dequant_int4(&q.storage(), &sc.storage(), k, n, group), vec![k, n]);
            vec![Some(g.matmul_nt(&w))]
        })
    }

    // ------------------------------------------------------------------
    // indexing
    // ------------------------------------------------------------------

    /// Gather rows of a 2D table: `[rows, cols]` -> `[n, cols]` (embedding lookup).
    pub fn index_select(&self, idx: impl IntoIndices) -> Tensor {
        assert_eq!(self.rank(), 2, "index_select expects a 2D table");
        let idx = idx.into_indices(self.device());
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let s = self.backend().index_select(&self.storage(), cols, &idx.storage());
        let n = idx.numel();
        Tensor::from_op(s, vec![n, cols], vec![self.clone()], "index_select", move |g| vec![Some(g.index_add_rows(rows, &idx))])
    }

    /// Sum rows of `self` `[n, cols]` into a `[rows, cols]` zero table at
    /// `idx` (the adjoint of [`Tensor::index_select`]; differentiable).
    fn index_add_rows(&self, rows: usize, idx: &Tensor) -> Tensor {
        let cols = self.dim(-1);
        let s = self.backend().index_add(&self.storage(), rows, cols, &idx.storage());
        let idx = idx.clone();
        Tensor::from_op(s, vec![rows, cols], vec![self.clone()], "index_add", move |g| vec![Some(g.index_select(&idx))])
    }

    /// `out[.., j, ..] = self[.., idx[.., j, ..], ..]` along `dim`; `idx` has the
    /// same shape as `self` except along `dim` (like `torch.gather`).
    pub fn gather(&self, dim: isize, idx: &Tensor) -> Tensor {
        let ax = norm_dim(dim, self.rank());
        let idx = idx.into_indices(self.device());
        assert_eq!(idx.rank(), self.rank(), "gather: index rank must match input rank");
        for (d, (&a, &b)) in self.shape().iter().zip(idx.shape()).enumerate() {
            assert!(d == ax || a == b, "gather: index shape {:?} incompatible with {:?}", idx.shape(), self.shape());
        }
        let idx_len = idx.shape()[ax];
        let s = self.backend().gather(&self.storage(), self.shape(), ax, &idx.storage(), idx_len);
        let in_shape = self.shape().to_vec();
        let out_shape = idx.shape().to_vec();
        Tensor::from_op(s, out_shape, vec![self.clone()], "gather", move |g| vec![Some(g.scatter_add_to(&idx, &in_shape, ax))])
    }

    /// Add `self` (shaped like `idx`) into a zero tensor of `shape` at `idx`
    /// along `ax` (the adjoint of [`Tensor::gather`]; differentiable).
    fn scatter_add_to(&self, idx: &Tensor, shape: &[usize], ax: usize) -> Tensor {
        let idx_len = idx.shape()[ax];
        let s = self.backend().scatter_add(&self.storage(), &idx.storage(), idx_len, shape, ax);
        let idx = idx.clone();
        Tensor::from_op(s, shape.to_vec(), vec![self.clone()], "scatter_add", move |g| vec![Some(g.gather(ax as isize, &idx))])
    }

    /// Elements `start..start+len` along `dim` (a view: only the offset changes).
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Tensor {
        let ax = norm_dim(dim, self.rank());
        let size = self.shape()[ax];
        assert!(start + len <= size, "narrow: {start}+{len} out of range for size {size}");
        if start == 0 && len == size {
            return self.clone();
        }
        let (_, st, off) = self.strided_parts();
        let mut out_shape = self.shape().to_vec();
        out_shape[ax] = len;
        let in_shape = self.shape().to_vec();
        let offset = off + start * st[ax];
        Tensor::view_op(self, out_shape, st, offset, "narrow", move |g| {
            // pad the gradient back with zeros on both sides
            let mut parts = Vec::new();
            let mut zshape = in_shape.clone();
            if start > 0 {
                zshape[ax] = start;
                parts.push(Tensor::zeros(&zshape, g.device()).to_dtype(g.dtype()));
            }
            parts.push(g.clone());
            if start + len < size {
                zshape[ax] = size - start - len;
                parts.push(Tensor::zeros(&zshape, g.device()).to_dtype(g.dtype()));
            }
            vec![Some(Tensor::cat(&parts.iter().collect::<Vec<_>>(), ax as isize))]
        })
    }

    /// Index `i` along `dim`, removing that dim.
    pub fn select(&self, dim: isize, i: usize) -> Tensor {
        self.narrow(dim, i, 1).squeeze(dim)
    }

    /// Concatenate along `dim` (like `torch.cat`).
    pub fn cat(xs: &[&Tensor], dim: isize) -> Tensor {
        assert!(!xs.is_empty(), "cat: empty list");
        let ax = norm_dim(dim, xs[0].rank());
        for x in xs {
            same_device(xs[0], x);
            assert_eq!(x.rank(), xs[0].rank(), "cat: rank mismatch");
            for d in 0..x.rank() {
                assert!(d == ax || x.shape()[d] == xs[0].shape()[d], "cat: shapes {:?} vs {:?}", x.shape(), xs[0].shape());
            }
        }
        if xs.len() == 1 {
            return xs[0].clone();
        }
        // mixed dtypes: cast every input to the common (promoted) dtype
        let dtype = xs.iter().skip(1).fold(xs[0].dtype(), |d, x| DType::promote(d, false, x.dtype(), false));
        if xs.iter().any(|x| x.dtype() != dtype) {
            let cast: Vec<Tensor> = xs.iter().map(|x| x.to_dtype(dtype)).collect();
            return Tensor::cat(&cast.iter().collect::<Vec<_>>(), dim);
        }
        let storages: Vec<_> = xs.iter().map(|x| x.storage()).collect();
        let srefs: Vec<_> = storages.iter().collect();
        let shapes: Vec<&[usize]> = xs.iter().map(|x| x.shape()).collect();
        let s = xs[0].backend().cat(&srefs, &shapes, ax);
        let mut out_shape = xs[0].shape().to_vec();
        out_shape[ax] = xs.iter().map(|x| x.shape()[ax]).sum();
        let sizes: Vec<usize> = xs.iter().map(|x| x.shape()[ax]).collect();
        let parents: Vec<Tensor> = xs.iter().map(|&x| x.clone()).collect();
        Tensor::from_op(s, out_shape, parents, "cat", move |g| {
            let mut off = 0;
            sizes
                .iter()
                .map(|&n| {
                    let part = g.narrow(ax as isize, off, n);
                    off += n;
                    Some(part)
                })
                .collect()
        })
    }

    /// Stack along a new dim (like `torch.stack`).
    pub fn stack(xs: &[&Tensor], dim: isize) -> Tensor {
        let us: Vec<Tensor> = xs.iter().map(|x| x.unsqueeze(dim)).collect();
        Tensor::cat(&us.iter().collect::<Vec<_>>(), dim)
    }

    // ------------------------------------------------------------------
    // fused ops
    // ------------------------------------------------------------------

    /// Run a numerically sensitive op in f32 when `self` is half precision.
    /// The result is cast back to the input dtype, except inside autocast
    /// (where these ops deliberately produce f32).
    fn in_f32(&self, f: impl FnOnce(&Tensor) -> Tensor) -> Tensor {
        if !self.dtype().is_half() {
            return f(self);
        }
        let y = f(&self.to_dtype(DType::F32));
        if autocast_dtype().is_some() {
            y
        } else {
            y.to_dtype(self.dtype())
        }
    }

    /// Softmax along `dim` (fused backend kernel; f32 math for half inputs).
    pub fn softmax(&self, dim: isize) -> Tensor {
        assert_float(self, "softmax");
        if self.dtype().is_half() {
            return self.in_f32(|x| x.softmax(dim));
        }
        let ax = norm_dim(dim, self.rank());
        let shape = self.shape().to_vec();
        let s = self.backend().softmax(&self.storage(), &shape, ax);
        let y = Tensor::from_storage(s.clone(), shape.clone());
        let x = self.clone();
        Tensor::from_op(s, shape.clone(), vec![self.clone()], "softmax", move |g| {
            if graph() {
                // dx = y (g - Σ g y), with y recomputed differentiably
                let y = x.softmax(ax as isize);
                return vec![Some(&y * &(g - &(g * &y).sum(ax as isize, true)))];
            }
            let s = g.backend().softmax_backward(&y.storage(), &g.storage(), &shape, ax);
            vec![Some(Tensor::from_storage(s, shape.clone()))]
        })
    }

    /// Log-softmax along `dim` (fused backend kernel; f32 math for half inputs).
    pub fn log_softmax(&self, dim: isize) -> Tensor {
        assert_float(self, "log_softmax");
        if self.dtype().is_half() {
            return self.in_f32(|x| x.log_softmax(dim));
        }
        let ax = norm_dim(dim, self.rank());
        let shape = self.shape().to_vec();
        let s = self.backend().log_softmax(&self.storage(), &shape, ax);
        let y = Tensor::from_storage(s.clone(), shape.clone());
        let x = self.clone();
        Tensor::from_op(s, shape.clone(), vec![self.clone()], "log_softmax", move |g| {
            if graph() {
                // dx = g - softmax(x) Σ g
                return vec![Some(g - &(x.softmax(ax as isize) * g.sum(ax as isize, true)))];
            }
            let s = g.backend().log_softmax_backward(&y.storage(), &g.storage(), &shape, ax);
            vec![Some(Tensor::from_storage(s, shape.clone()))]
        })
    }

    /// Normalize over the last dim to zero mean / unit variance (no affine;
    /// f32 math for half inputs).
    pub fn normalize_last(&self, eps: f32) -> Tensor {
        assert_float(self, "layer_norm");
        if self.dtype().is_half() {
            return self.in_f32(|x| x.normalize_last(eps));
        }
        let dim = self.dim(-1);
        let rows = self.numel() / dim.max(1);
        let (xh, rstd) = self.backend().layer_norm(&self.storage(), rows, dim, eps);
        let shape = self.shape().to_vec();
        let xhat = Tensor::from_storage(xh.clone(), shape.clone());
        let x = self.clone();
        Tensor::from_op(xh, shape.clone(), vec![self.clone()], "layer_norm", move |g| {
            if graph() {
                // dx = rstd (g - mean(g) - x̂ mean(g x̂)), from differentiable ops
                let xc = &x - &x.mean(-1, true);
                let rstd = (&xc * &xc).mean(-1, true).add_scalar(eps).sqrt().recip();
                let xh = &xc * &rstd;
                let inner = g - &g.mean(-1, true) - &xh * &(g * &xh).mean(-1, true);
                return vec![Some(rstd * inner)];
            }
            let s = g.backend().layer_norm_backward(&g.storage(), &xhat.storage(), &rstd, rows, dim);
            vec![Some(Tensor::from_storage(s, shape.clone()))]
        })
    }

    // ------------------------------------------------------------------
    // convolution helpers
    // ------------------------------------------------------------------

    /// `[N, C, H, W]` -> `[N, C*KH*KW, OH*OW]` patch matrix.
    pub fn im2col(&self, kh: usize, kw: usize, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.rank(), 4, "im2col expects [N, C, H, W]");
        let sh = self.shape();
        let g = Conv2dGeom { n: sh[0], c: sh[1], h: sh[2], w: sh[3], kh, kw, stride, padding };
        let (oh, ow) = g.out_hw();
        let s = self.backend().im2col(&self.storage(), g);
        let in_shape = sh.to_vec();
        Tensor::from_op(s, vec![g.n, g.c * kh * kw, oh * ow], vec![self.clone()], "im2col", move |grad| vec![Some(grad.col2im(g, &in_shape))])
    }

    /// Adjoint of [`Tensor::im2col`]: sum patch columns back into an image
    /// `[N, C, H, W]` (differentiable).
    fn col2im(&self, g: Conv2dGeom, shape: &[usize]) -> Tensor {
        let s = self.backend().col2im(&self.storage(), g);
        Tensor::from_op(s, shape.to_vec(), vec![self.clone()], "col2im", move |grad| {
            vec![Some(grad.im2col(g.kh, g.kw, g.stride, g.padding))]
        })
    }

    // ------------------------------------------------------------------
    // devices
    // ------------------------------------------------------------------

    /// Copy to another device (differentiable).
    pub fn to(&self, device: Device) -> Tensor {
        let src = self.device();
        if src == device {
            return self.clone();
        }
        let s = self.storage().to_device(device);
        Tensor::from_op(s, self.shape().to_vec(), vec![self.clone()], "to_device", move |g| vec![Some(g.to(src))])
    }
}

// ----------------------------------------------------------------------
// operator overloading: &a + &b, a * 2.0, -a, ...
// ----------------------------------------------------------------------

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $func:ident) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                self.$func(rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                (&self).$func(&rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                (&self).$func(rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                self.$func(&rhs)
            }
        }
        impl std::ops::$trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                self.$func(&Tensor::scalar(rhs, self.device()))
            }
        }
        impl std::ops::$trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                (&self).$func(&Tensor::scalar(rhs, self.device()))
            }
        }
    };
}

impl_binop!(Add, add, add);
impl_binop!(Sub, sub, sub);
impl_binop!(Mul, mul, mul);
impl_binop!(Div, div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
