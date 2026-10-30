//! Differentiable tensor operations. Each op computes its forward result with
//! backend primitives and registers a backward closure built from other ops.

use crate::backend::{BinaryOp, Conv2dGeom, MatmulSpec, ReduceOp, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_shape, norm_dim, numel};
use crate::tensor::Tensor;

fn need(t: &Tensor) -> bool {
    t.requires_grad()
}

fn same_device(a: &Tensor, b: &Tensor) {
    assert_eq!(a.device(), b.device(), "tensors are on different devices ({} vs {})", a.device(), b.device());
}

impl Tensor {
    // ------------------------------------------------------------------
    // elementwise unary
    // ------------------------------------------------------------------

    fn unary_raw(&self, op: UnaryOp) -> Tensor {
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
        let out_shape = broadcast_shape(self.shape(), other.shape());
        let s = self.backend().binary(op, &self.storage(), self.shape(), &other.storage(), other.shape(), &out_shape);
        Tensor::from_storage(s, out_shape)
    }

    fn binary_op<F>(&self, op: BinaryOp, other: &Tensor, name: &'static str, backward: F) -> Tensor
    where
        F: Fn(&Tensor, &Tensor, &Tensor) -> (Option<Tensor>, Option<Tensor>) + Send + Sync + 'static,
    {
        let out = self.binary_raw(op, other);
        let (a, b) = (self.clone(), other.clone());
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
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let out = self.matmul_raw(other, false, false);
        let (a, b) = (self.clone(), other.clone());
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

    /// Gather rows of a 2D tensor: `[rows, cols]` -> `[idx.len(), cols]`.
    pub fn index_select(&self, idx: &[usize]) -> Tensor {
        assert_eq!(self.rank(), 2, "index_select expects a 2D table");
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let s = self.backend().index_select(&self.storage(), cols, idx);
        let idx = idx.to_vec();
        Tensor::from_op(s, vec![idx.len(), cols], vec![self.clone()], "index_select", move |g| {
            let s = g.backend().index_add(&g.storage(), rows, cols, &idx);
            vec![Some(Tensor::from_storage(s, vec![rows, cols]))]
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
