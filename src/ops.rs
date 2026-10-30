//! Differentiable tensor operations. Each op computes its forward result with
//! backend primitives and registers a backward closure built from other ops.

use crate::backend::{BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_shape, norm_dim, numel};
use crate::tensor::{autocast_dtype, Tensor};

fn need(t: &Tensor) -> bool {
    t.requires_grad()
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

    fn unary_op<F>(&self, op: UnaryOp, name: &'static str, backward: F) -> Tensor
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Send + Sync + 'static,
    {
        let out = self.unary_raw(op);
        let x = self.clone();
        let y = out.clone(); // detached copy of the output for the backward pass
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![self.clone()], name, move |g| {
            vec![Some(backward(g, &x, &y))]
        })
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(UnaryOp::Neg, "neg", |g, _, _| g.neg())
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(UnaryOp::Exp, "exp", |g, _, y| g * y)
    }

    pub fn log(&self) -> Tensor {
        self.unary_op(UnaryOp::Log, "log", |g, x, _| g / x)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(UnaryOp::Sqrt, "sqrt", |g, _, y| g * &(y.unary_raw(UnaryOp::Recip) * 0.5))
    }

    pub fn recip(&self) -> Tensor {
        self.unary_op(UnaryOp::Recip, "recip", |g, _, y| (g * &(y * y)).neg())
    }

    pub fn abs(&self) -> Tensor {
        self.unary_op(UnaryOp::Abs, "abs", |g, x, _| {
            let sign = &x.gt_scalar(0.0) - &x.lt_scalar(0.0);
            g * &sign
        })
    }

    pub fn tanh(&self) -> Tensor {
        self.unary_op(UnaryOp::Tanh, "tanh", |g, _, y| g * &((y * y).neg() + 1.0))
    }

    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(UnaryOp::Sigmoid, "sigmoid", |g, _, y| g * &(y * &(y.neg() + 1.0)))
    }

    pub fn relu(&self) -> Tensor {
        self.unary_op(UnaryOp::Relu, "relu", |g, x, _| g * &x.gt_scalar(0.0))
    }

    /// GELU (tanh approximation), as used in GPT/BERT.
    pub fn gelu(&self) -> Tensor {
        self.unary_op(UnaryOp::Gelu, "gelu", |g, x, _| g * &x.unary_raw(UnaryOp::GeluGrad))
    }

    pub fn sin(&self) -> Tensor {
        self.unary_op(UnaryOp::Sin, "sin", |g, x, _| g * &x.unary_raw(UnaryOp::Cos))
    }

    pub fn cos(&self) -> Tensor {
        self.unary_op(UnaryOp::Cos, "cos", |g, x, _| (g * &x.unary_raw(UnaryOp::Sin)).neg())
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

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Add, other, "add", |g, a, b| {
            (need(a).then(|| g.clone()), need(b).then(|| g.clone()))
        })
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Sub, other, "sub", |g, a, b| (need(a).then(|| g.clone()), need(b).then(|| g.neg())))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op(BinaryOp::Mul, other, "mul", |g, a, b| (need(a).then(|| g * b), need(b).then(|| g * a)))
    }

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
            let d = x.binary_raw(BinaryOp::Pow, &Tensor::scalar(p - 1.0, x.device())) * p;
            vec![Some(g * &d)]
        })
    }

    fn scalar_like(&self, v: f32) -> Tensor {
        Tensor::scalar(v, self.device())
    }

    pub fn add_scalar(&self, v: f32) -> Tensor {
        self.add(&self.scalar_like(v))
    }

    pub fn mul_scalar(&self, v: f32) -> Tensor {
        self.mul(&self.scalar_like(v))
    }

    // comparisons (not differentiable; return 0/1 tensors)
    pub fn eq_t(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Eq, other)
    }

    pub fn gt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Gt, other)
    }

    pub fn lt(&self, other: &Tensor) -> Tensor {
        self.binary_raw(BinaryOp::Lt, other)
    }

    pub fn gt_scalar(&self, v: f32) -> Tensor {
        self.gt(&self.scalar_like(v))
    }

    pub fn lt_scalar(&self, v: f32) -> Tensor {
        self.lt(&self.scalar_like(v))
    }

    // ------------------------------------------------------------------
    // broadcasting helpers
    // ------------------------------------------------------------------

    /// Broadcast to `shape` (materialized copy).
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        if self.shape() == shape {
            return self.clone();
        }
        assert_eq!(broadcast_shape(self.shape(), shape), shape, "cannot expand {:?} to {shape:?}", self.shape());
        let s = self.backend().expand(&self.storage(), self.shape(), shape);
        let in_shape = self.shape().to_vec();
        Tensor::from_op(s, shape.to_vec(), vec![self.clone()], "expand", move |g| vec![Some(g.sum_to(&in_shape))])
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

    pub fn mean(&self, dim: isize, keepdim: bool) -> Tensor {
        let n = self.dim(dim) as f32;
        self.sum(dim, keepdim) * (1.0 / n)
    }

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

    pub fn min(&self, dim: isize, keepdim: bool) -> Tensor {
        self.neg().max(dim, keepdim).neg()
    }

    /// Index of the max along `dim` (as f32, not differentiable).
    pub fn argmax(&self, dim: isize, keepdim: bool) -> Tensor {
        let (ax, keep, drop) = self.reduced_shapes(dim);
        let s = self.backend().reduce(ReduceOp::ArgMax, &self.storage(), self.shape(), ax);
        Tensor::from_storage(s, if keepdim { keep } else { drop })
    }

    // ------------------------------------------------------------------
    // shape manipulation
    // ------------------------------------------------------------------

    /// Reshape (shares storage; all tensors are contiguous).
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(numel(shape), self.numel(), "cannot reshape {:?} to {shape:?}", self.shape());
        if shape == self.shape() {
            return self.clone();
        }
        let in_shape = self.shape().to_vec();
        Tensor::from_op(self.storage(), shape.to_vec(), vec![self.clone()], "reshape", move |g| {
            vec![Some(g.reshape(&in_shape))]
        })
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

    pub fn flatten(&self, start_dim: usize) -> Tensor {
        let mut s = self.shape()[..start_dim].to_vec();
        s.push(self.shape()[start_dim..].iter().product());
        self.reshape(&s)
    }

    pub fn unsqueeze(&self, dim: isize) -> Tensor {
        let d = norm_dim(dim, self.rank() + 1);
        let mut s = self.shape().to_vec();
        s.insert(d, 1);
        self.reshape(&s)
    }

    pub fn squeeze(&self, dim: isize) -> Tensor {
        let d = norm_dim(dim, self.rank());
        assert_eq!(self.shape()[d], 1, "squeeze: dim {d} has size {}", self.shape()[d]);
        let mut s = self.shape().to_vec();
        s.remove(d);
        self.reshape(&s)
    }

    pub fn permute(&self, perm: &[usize]) -> Tensor {
        assert_eq!(perm.len(), self.rank(), "permute: wrong number of dims");
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return self.clone();
        }
        let s = self.backend().permute(&self.storage(), self.shape(), perm);
        let out_shape: Vec<usize> = perm.iter().map(|&p| self.shape()[p]).collect();
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        Tensor::from_op(s, out_shape, vec![self.clone()], "permute", move |g| vec![Some(g.permute(&inv))])
    }

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

    /// Batched matmul without autograd, with optional transposes of the last two dims.
    /// Supports: 2D @ 2D, ND @ 2D, 2D @ ND, and ND @ ND with equal batch dims.
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

        let (spec, batch_shape) = if bbatch.is_empty() && !ta && !abatch.is_empty() {
            // fold A's batch dims into its rows: one big GEMM
            let rows = numel(abatch) * m;
            let spec = MatmulSpec { batch: 1, m: rows, k, n, trans_a: false, trans_b: tb, a_batch_stride: 0, b_batch_stride: 0 };
            (spec, abatch.to_vec())
        } else if abatch == bbatch {
            let b = numel(abatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: m * k, b_batch_stride: k * n };
            (spec, abatch.to_vec())
        } else if abatch.is_empty() {
            let b = numel(bbatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: 0, b_batch_stride: k * n };
            (spec, bbatch.to_vec())
        } else if bbatch.is_empty() {
            let b = numel(abatch);
            let spec = MatmulSpec { batch: b, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: m * k, b_batch_stride: 0 };
            (spec, abatch.to_vec())
        } else {
            panic!("matmul: unsupported batch broadcast {ash:?} @ {bsh:?}");
        };
        let s = self.backend().matmul(&self.storage(), &other.storage(), spec);
        let mut out_shape = batch_shape;
        out_shape.extend([m, n]);
        Tensor::from_storage(s, out_shape)
    }

    /// Matrix product of the last two dims, batched over leading dims.
    /// Inside [`crate::autocast`] both operands are cast to the autocast dtype.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (a, b) = match autocast_dtype() {
            Some(d) if self.dtype().is_float() && other.dtype().is_float() => (self.to_dtype(d), other.to_dtype(d)),
            _ => promoted(self, other),
        };
        let out = a.matmul_raw(&b, false, false);
        Tensor::from_op(out.storage(), out.shape().to_vec(), vec![a.clone(), b.clone()], "matmul", move |g| {
            let ga = need(&a).then(|| g.matmul_raw(&b, false, true).sum_to(a.shape()));
            let gb = need(&b).then(|| {
                if b.rank() == 2 && a.rank() > 2 {
                    let k = a.dim(-1);
                    let n = g.dim(-1);
                    let af = a.reshape(&[a.numel() / k, k]);
                    let gf = g.reshape(&[g.numel() / n, n]);
                    af.matmul_raw(&gf, true, false)
                } else {
                    a.matmul_raw(g, true, false).sum_to(b.shape())
                }
            });
            vec![ga, gb]
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
        Tensor::from_op(s, vec![n, cols], vec![self.clone()], "index_select", move |g| {
            let s = g.backend().index_add(&g.storage(), rows, cols, &idx.storage());
            vec![Some(Tensor::from_storage(s, vec![rows, cols]))]
        })
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
        Tensor::from_op(s, out_shape, vec![self.clone()], "gather", move |g| {
            let s = g.backend().scatter_add(&g.storage(), &idx.storage(), idx_len, &in_shape, ax);
            vec![Some(Tensor::from_storage(s, in_shape.clone()))]
        })
    }

    /// Elements `start..start+len` along `dim` (copy).
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Tensor {
        let ax = norm_dim(dim, self.rank());
        let size = self.shape()[ax];
        assert!(start + len <= size, "narrow: {start}+{len} out of range for size {size}");
        if start == 0 && len == size {
            return self.clone();
        }
        let s = self.backend().narrow(&self.storage(), self.shape(), ax, start, len);
        let mut out_shape = self.shape().to_vec();
        out_shape[ax] = len;
        let in_shape = self.shape().to_vec();
        Tensor::from_op(s, out_shape, vec![self.clone()], "narrow", move |g| {
            // pad the gradient back with zeros on both sides
            let mut parts = Vec::new();
            let mut zshape = in_shape.clone();
            if start > 0 {
                zshape[ax] = start;
                parts.push(Tensor::zeros(&zshape, g.device()));
            }
            parts.push(g.clone());
            if start + len < size {
                zshape[ax] = size - start - len;
                parts.push(Tensor::zeros(&zshape, g.device()));
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
        Tensor::from_op(s, shape.clone(), vec![self.clone()], "softmax", move |g| {
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
        Tensor::from_op(s, shape.clone(), vec![self.clone()], "log_softmax", move |g| {
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
        Tensor::from_op(xh, shape.clone(), vec![self.clone()], "layer_norm", move |g| {
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
        Tensor::from_op(s, vec![g.n, g.c * kh * kw, oh * ow], vec![self.clone()], "im2col", move |grad| {
            let s = grad.backend().col2im(&grad.storage(), g);
            vec![Some(Tensor::from_storage(s, in_shape.clone()))]
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
