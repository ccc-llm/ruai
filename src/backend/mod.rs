//! Backend abstraction.
//!
//! Every tensor operation in the engine (including autograd and all `nn` layers)
//! is built from the *primitives* in the [`Backend`] trait. All storages are
//! contiguous, row-major buffers tagged with a [`DType`].
//!
//! To add a new device (CUDA, MPS, ...) you only need to:
//!   1. add a variant to [`Storage`],
//!   2. implement the required methods of [`Backend`] for it,
//!   3. return it from [`backend_for`].
//!
//! The *fused* primitives (`softmax`, `log_softmax`, `layer_norm`, ...) have
//! default implementations composed from the required ones, so a new backend
//! works as soon as the required methods exist and can override the fused ones
//! later for speed.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "mps")]
pub mod mps;

use crate::device::Device;
use std::fmt;
use std::sync::Arc;

/// Element type of a storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float: all differentiable math.
    F32,
    /// 32-bit signed int: indices (embedding ids, class targets, argmax results).
    I32,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        4
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::I32 => write!(f, "i32"),
        }
    }
}

/// Host-memory buffer of one dtype.
#[derive(Clone)]
pub enum CpuStorage {
    F32(Arc<Vec<f32>>),
    I32(Arc<Vec<i32>>),
}

/// Device-resident buffer. Cheap to clone (reference counted); storages are
/// never mutated in place once created.
#[derive(Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaStorage),
    #[cfg(feature = "mps")]
    Mps(mps::MpsStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => Device::Cuda(s.ordinal),
            #[cfg(feature = "mps")]
            Storage::Mps(_) => Device::Mps,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(CpuStorage::F32(_)) => DType::F32,
            Storage::Cpu(CpuStorage::I32(_)) => DType::I32,
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => s.dtype,
            #[cfg(feature = "mps")]
            Storage::Mps(s) => s.dtype,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(CpuStorage::F32(v)) => v.len(),
            Storage::Cpu(CpuStorage::I32(v)) => v.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => s.len,
            #[cfg(feature = "mps")]
            Storage::Mps(s) => s.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn backend(&self) -> &'static dyn Backend {
        backend_for(self.device())
    }

    /// Copy to host as `f32` (integers are converted).
    pub fn to_vec(&self) -> Vec<f32> {
        self.backend().to_f32(self)
    }

    /// Copy to host as `i32` (floats are truncated).
    pub fn to_vec_i32(&self) -> Vec<i32> {
        self.backend().to_i32(self)
    }

    /// Move/copy the storage to another device (through host memory), keeping its dtype.
    pub fn to_device(&self, device: Device) -> Storage {
        if self.device() == device {
            return self.clone();
        }
        let be = backend_for(device);
        match self.dtype() {
            DType::F32 => be.from_f32(self.to_vec()),
            DType::I32 => be.from_i32(self.to_vec_i32()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Recip,
    Abs,
    Tanh,
    Sigmoid,
    Relu,
    /// GELU (tanh approximation)
    Gelu,
    /// d/dx GELU (tanh approximation)
    GeluGrad,
    Sin,
    Cos,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    /// 1.0 where equal, else 0.0
    Eq,
    /// 1.0 where a > b, else 0.0
    Gt,
    /// 1.0 where a < b, else 0.0
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    /// Index of the maximum along the axis (I32 output).
    ArgMax,
}

/// Parameters for im2col / col2im (2D convolution lowering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dGeom {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2dGeom {
    pub fn out_hw(&self) -> (usize, usize) {
        let oh = (self.h + 2 * self.padding - self.kh) / self.stride + 1;
        let ow = (self.w + 2 * self.padding - self.kw) / self.stride + 1;
        (oh, ow)
    }
}

/// Batched GEMM description: for each batch `i`,
/// `C[i] (m x n) = op(A[i]) (m x k) @ op(B[i]) (k x n)`,
/// where `op` optionally transposes. `A[i]` starts at `i * a_batch_stride`
/// (0 = broadcast the same matrix), likewise for `B`. `C` is contiguous.
#[derive(Clone, Copy, Debug)]
pub struct MatmulSpec {
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a_batch_stride: usize,
    pub b_batch_stride: usize,
}

/// Hyper-parameters of one Adam/AdamW update (bias corrections precomputed).
#[derive(Clone, Copy, Debug)]
pub struct AdamParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// AdamW-style decay applied to the weights instead of the gradient.
    pub decoupled: bool,
    /// `1 - beta1^t`
    pub bias_correction1: f32,
    /// `1 - beta2^t`
    pub bias_correction2: f32,
}

fn keep_axis(shape: &[usize], axis: usize) -> Vec<usize> {
    let mut k = shape.to_vec();
    k[axis] = 1;
    k
}

/// The operations a device must provide.
///
/// Shapes are passed explicitly; all inputs and outputs are contiguous.
/// Unless stated otherwise, inputs and outputs are `F32`; index arguments are
/// `I32` storages on the same device. Each method maps closely to a standard
/// GPU kernel / library call (cuBLAS / MPSMatrixMultiplication for `matmul`,
/// simple elementwise kernels for `unary`/`binary`, etc.).
#[allow(clippy::wrong_self_convention)]
pub trait Backend: Send + Sync {
    fn device(&self) -> Device;

    /// Block until all queued work on this device has finished
    /// (no-op for synchronous backends such as the CPU).
    fn synchronize(&self) {}

    // ---- memory ----
    fn from_f32(&self, data: Vec<f32>) -> Storage;
    fn from_i32(&self, data: Vec<i32>) -> Storage;
    /// Copy to host as f32 (any dtype).
    fn to_f32(&self, x: &Storage) -> Vec<f32>;
    /// Copy to host as i32 (any dtype).
    fn to_i32(&self, x: &Storage) -> Vec<i32>;
    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage;
    fn cast(&self, x: &Storage, dtype: DType) -> Storage;

    // ---- elementwise ----
    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage;
    /// Elementwise binary op with NumPy broadcasting into `out_shape`.
    /// Comparison ops also accept I32 inputs (output is always F32).
    fn binary(
        &self,
        op: BinaryOp,
        a: &Storage,
        a_shape: &[usize],
        b: &Storage,
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> Storage;
    /// Broadcast `x` (of `shape`) to `out_shape`, materializing a copy.
    fn expand(&self, x: &Storage, shape: &[usize], out_shape: &[usize]) -> Storage;
    /// `cond != 0 ? a : b`, all three of the same length.
    fn where_cond(&self, cond: &Storage, a: &Storage, b: &Storage) -> Storage;

    // ---- reductions ----
    /// Reduce over `axis` of `shape`; output has that axis removed.
    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage;

    // ---- linear algebra / layout ----
    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage;
    /// Generic axis permutation (copy).
    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage;
    /// Elements `start..start+len` along `axis` (any dtype).
    fn narrow(&self, x: &Storage, shape: &[usize], axis: usize, start: usize, len: usize) -> Storage;
    /// Concatenate along `axis`; all shapes agree except on `axis` (any dtype).
    fn cat(&self, xs: &[&Storage], shapes: &[&[usize]], axis: usize) -> Storage;

    // ---- indexing ----
    /// Gather rows: `table` is `[rows, cols]`, output `[idx.len(), cols]`.
    fn index_select(&self, table: &Storage, cols: usize, idx: &Storage) -> Storage;
    /// Scatter-add rows of `src` (`[idx.len(), cols]`) into a zero `[rows, cols]`.
    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &Storage) -> Storage;
    /// `out[o, j, i] = x[o, idx[o, j, i], i]` where `x` is `(outer, len, inner)`
    /// around `axis` and `idx` is `(outer, idx_len, inner)`.
    fn gather(&self, x: &Storage, shape: &[usize], axis: usize, idx: &Storage, idx_len: usize) -> Storage;
    /// Adjoint of `gather`: scatter-add `src` (shaped like `idx`) into zeros of `shape`.
    fn scatter_add(&self, src: &Storage, idx: &Storage, idx_len: usize, shape: &[usize], axis: usize) -> Storage;

    // ---- convolution lowering ----
    /// `[N, C, H, W]` -> `[N, C*KH*KW, OH*OW]`
    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage;
    /// Adjoint of `im2col`: `[N, C*KH*KW, OH*OW]` -> `[N, C, H, W]` (accumulating).
    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage;

    // ---- fused ops (default: composed from the primitives above) ----

    /// Softmax along `axis`.
    fn softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let keep = keep_axis(shape, axis);
        let m = self.reduce(ReduceOp::Max, x, shape, axis);
        let d = self.binary(BinaryOp::Sub, x, shape, &m, &keep, shape);
        let e = self.unary(UnaryOp::Exp, &d);
        let s = self.reduce(ReduceOp::Sum, &e, shape, axis);
        self.binary(BinaryOp::Div, &e, shape, &s, &keep, shape)
    }

    /// Given `y = softmax(x)` and `g = dL/dy`: `dL/dx = y * (g - sum(g * y))`.
    fn softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        let keep = keep_axis(shape, axis);
        let gy = self.binary(BinaryOp::Mul, g, shape, y, shape, shape);
        let s = self.reduce(ReduceOp::Sum, &gy, shape, axis);
        let d = self.binary(BinaryOp::Sub, g, shape, &s, &keep, shape);
        self.binary(BinaryOp::Mul, y, shape, &d, shape, shape)
    }

    /// Log-softmax along `axis`.
    fn log_softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let keep = keep_axis(shape, axis);
        let m = self.reduce(ReduceOp::Max, x, shape, axis);
        let d = self.binary(BinaryOp::Sub, x, shape, &m, &keep, shape);
        let e = self.unary(UnaryOp::Exp, &d);
        let s = self.unary(UnaryOp::Log, &self.reduce(ReduceOp::Sum, &e, shape, axis));
        self.binary(BinaryOp::Sub, &d, shape, &s, &keep, shape)
    }

    /// Given `y = log_softmax(x)` and `g`: `dL/dx = g - exp(y) * sum(g)`.
    fn log_softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        let keep = keep_axis(shape, axis);
        let s = self.reduce(ReduceOp::Sum, g, shape, axis);
        let p = self.unary(UnaryOp::Exp, y);
        let ps = self.binary(BinaryOp::Mul, &p, shape, &s, &keep, shape);
        self.binary(BinaryOp::Sub, g, shape, &ps, shape, shape)
    }

    /// Normalize each row of `[rows, dim]`: returns `(x_hat, rstd)` with
    /// `x_hat = (x - mean) * rstd`, `rstd = 1/sqrt(var + eps)` of shape `[rows]`.
    fn layer_norm(&self, x: &Storage, rows: usize, dim: usize, eps: f32) -> (Storage, Storage) {
        let shape = [rows, dim];
        let keep = [rows, 1];
        let inv_n = self.from_f32(vec![1.0 / dim as f32]);
        let mean = self.binary(BinaryOp::Mul, &self.reduce(ReduceOp::Sum, x, &shape, 1), &[rows], &inv_n, &[], &[rows]);
        let xc = self.binary(BinaryOp::Sub, x, &shape, &mean, &keep, &shape);
        let sq = self.binary(BinaryOp::Mul, &xc, &shape, &xc, &shape, &shape);
        let var = self.binary(BinaryOp::Mul, &self.reduce(ReduceOp::Sum, &sq, &shape, 1), &[rows], &inv_n, &[], &[rows]);
        let var_eps = self.binary(BinaryOp::Add, &var, &[rows], &self.from_f32(vec![eps]), &[], &[rows]);
        let rstd = self.unary(UnaryOp::Recip, &self.unary(UnaryOp::Sqrt, &var_eps));
        let xhat = self.binary(BinaryOp::Mul, &xc, &shape, &rstd, &keep, &shape);
        (xhat, rstd)
    }

    /// Backward of `layer_norm` w.r.t. `x`:
    /// `dx = rstd * (g - mean(g) - x_hat * mean(g * x_hat))`.
    fn layer_norm_backward(&self, g: &Storage, xhat: &Storage, rstd: &Storage, rows: usize, dim: usize) -> Storage {
        let shape = [rows, dim];
        let keep = [rows, 1];
        let inv_n = self.from_f32(vec![1.0 / dim as f32]);
        let mg = self.binary(BinaryOp::Mul, &self.reduce(ReduceOp::Sum, g, &shape, 1), &[rows], &inv_n, &[], &[rows]);
        let gx = self.binary(BinaryOp::Mul, g, &shape, xhat, &shape, &shape);
        let mgx = self.binary(BinaryOp::Mul, &self.reduce(ReduceOp::Sum, &gx, &shape, 1), &[rows], &inv_n, &[], &[rows]);
        let t = self.binary(BinaryOp::Sub, g, &shape, &mg, &keep, &shape);
        let xm = self.binary(BinaryOp::Mul, xhat, &shape, &mgx, &keep, &shape);
        let t = self.binary(BinaryOp::Sub, &t, &shape, &xm, &shape, &shape);
        self.binary(BinaryOp::Mul, &t, &shape, rstd, &keep, &shape)
    }

    /// One Adam/AdamW step on a flat parameter: returns `(new_param, new_m, new_v)`.
    fn adam_step(&self, p: &Storage, g: &Storage, m: &Storage, v: &Storage, h: AdamParams) -> (Storage, Storage, Storage) {
        let n = [p.len()];
        let sc = |x: f32| self.from_f32(vec![x]);
        let mul = |a: &Storage, s: f32| self.binary(BinaryOp::Mul, a, &n, &sc(s), &[], &n);
        let add = |a: &Storage, b: &Storage| self.binary(BinaryOp::Add, a, &n, b, &n, &n);
        let g = if h.weight_decay != 0.0 && !h.decoupled { add(g, &mul(p, h.weight_decay)) } else { g.clone() };
        let m = add(&mul(m, h.beta1), &mul(&g, 1.0 - h.beta1));
        let g2 = self.binary(BinaryOp::Mul, &g, &n, &g, &n, &n);
        let v = add(&mul(v, h.beta2), &mul(&g2, 1.0 - h.beta2));
        let denom = self.unary(UnaryOp::Sqrt, &mul(&v, 1.0 / h.bias_correction2));
        let denom = self.binary(BinaryOp::Add, &denom, &n, &sc(h.eps), &[], &n);
        let upd = self.binary(BinaryOp::Div, &mul(&m, 1.0 / h.bias_correction1), &n, &denom, &n, &n);
        let mut newp = self.binary(BinaryOp::Sub, p, &n, &mul(&upd, h.lr), &n, &n);
        if h.weight_decay != 0.0 && h.decoupled {
            newp = self.binary(BinaryOp::Sub, &newp, &n, &mul(p, h.lr * h.weight_decay), &n, &n);
        }
        (newp, m, v)
    }
}

static CPU: cpu::CpuBackend = cpu::CpuBackend;

/// Look up the backend for a device.
pub fn backend_for(device: Device) -> &'static dyn Backend {
    match device {
        Device::Cpu => &CPU,
        #[cfg(feature = "cuda")]
        Device::Cuda(i) => cuda::backend(i),
        #[cfg(feature = "mps")]
        Device::Mps => mps::backend(),
        #[allow(unreachable_patterns)]
        d => panic!("device {d} is not available in this build (enable the corresponding cargo feature)"),
    }
}

/// The default (composed) implementations of the fused ops, usable from tests
/// to check a backend's fused overrides against them.
pub mod reference {
    use super::*;

    /// A wrapper that forwards only the required primitives, so every fused op
    /// uses the trait's default implementation.
    pub struct Unfused(pub &'static dyn Backend);

    impl Backend for Unfused {
        fn device(&self) -> Device {
            self.0.device()
        }
        fn from_f32(&self, d: Vec<f32>) -> Storage {
            self.0.from_f32(d)
        }
        fn from_i32(&self, d: Vec<i32>) -> Storage {
            self.0.from_i32(d)
        }
        fn to_f32(&self, x: &Storage) -> Vec<f32> {
            self.0.to_f32(x)
        }
        fn to_i32(&self, x: &Storage) -> Vec<i32> {
            self.0.to_i32(x)
        }
        fn full(&self, len: usize, v: f32, dt: DType) -> Storage {
            self.0.full(len, v, dt)
        }
        fn cast(&self, x: &Storage, dt: DType) -> Storage {
            self.0.cast(x, dt)
        }
        fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
            self.0.unary(op, x)
        }
        fn binary(&self, op: BinaryOp, a: &Storage, sa: &[usize], b: &Storage, sb: &[usize], so: &[usize]) -> Storage {
            self.0.binary(op, a, sa, b, sb, so)
        }
        fn expand(&self, x: &Storage, s: &[usize], so: &[usize]) -> Storage {
            self.0.expand(x, s, so)
        }
        fn where_cond(&self, c: &Storage, a: &Storage, b: &Storage) -> Storage {
            self.0.where_cond(c, a, b)
        }
        fn reduce(&self, op: ReduceOp, x: &Storage, s: &[usize], axis: usize) -> Storage {
            self.0.reduce(op, x, s, axis)
        }
        fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
            self.0.matmul(a, b, spec)
        }
        fn permute(&self, x: &Storage, s: &[usize], p: &[usize]) -> Storage {
            self.0.permute(x, s, p)
        }
        fn narrow(&self, x: &Storage, s: &[usize], axis: usize, start: usize, len: usize) -> Storage {
            self.0.narrow(x, s, axis, start, len)
        }
        fn cat(&self, xs: &[&Storage], shapes: &[&[usize]], axis: usize) -> Storage {
            self.0.cat(xs, shapes, axis)
        }
        fn index_select(&self, t: &Storage, cols: usize, idx: &Storage) -> Storage {
            self.0.index_select(t, cols, idx)
        }
        fn index_add(&self, s: &Storage, rows: usize, cols: usize, idx: &Storage) -> Storage {
            self.0.index_add(s, rows, cols, idx)
        }
        fn gather(&self, x: &Storage, s: &[usize], axis: usize, idx: &Storage, l: usize) -> Storage {
            self.0.gather(x, s, axis, idx, l)
        }
        fn scatter_add(&self, src: &Storage, idx: &Storage, l: usize, s: &[usize], axis: usize) -> Storage {
            self.0.scatter_add(src, idx, l, s, axis)
        }
        fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
            self.0.im2col(x, g)
        }
        fn col2im(&self, c: &Storage, g: Conv2dGeom) -> Storage {
            self.0.col2im(c, g)
        }
    }
}
