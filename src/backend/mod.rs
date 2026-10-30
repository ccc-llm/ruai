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
#[cfg(any(feature = "mps", feature = "cuda"))]
mod kernel_names;
#[cfg(feature = "mps")]
pub mod mps;

use crate::device::Device;
use std::fmt;
use std::sync::Arc;

/// Element type of a storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float: the default for all differentiable math.
    F32,
    /// IEEE half precision (5-bit exponent): mixed-precision compute.
    F16,
    /// bfloat16 (8-bit exponent, f32 range): mixed-precision compute.
    BF16,
    /// 32-bit signed int: indices (embedding ids, class targets, argmax results).
    I32,
    /// 8-bit signed int: quantized weights (storage only; see `matmul_int8`).
    I8,
}

impl DType {
    /// Bytes per element.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::I8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
        }
    }

    /// `true` for `F32`, `F16` and `BF16`.
    pub fn is_float(&self) -> bool {
        !matches!(self, DType::I32 | DType::I8)
    }

    /// `true` for the 16-bit float types `F16` and `BF16`.
    pub fn is_half(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }

    /// Result type of a binary op between dtypes `a` and `b` (`a_scalar` /
    /// `b_scalar`: the operand is rank 0). Same dtype -> itself; int with
    /// float -> the float; two floats -> F32, except that a rank-0 float
    /// scalar adopts the other operand's dtype (so `half_tensor * 2.0` stays half).
    pub fn promote(a: DType, a_scalar: bool, b: DType, b_scalar: bool) -> DType {
        if a == b {
            return a;
        }
        match (a.is_float(), b.is_float()) {
            (true, false) => a,
            (false, true) => b,
            _ if a_scalar && !b_scalar => b,
            _ if b_scalar && !a_scalar => a,
            _ => DType::F32,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I8 => write!(f, "i8"),
        }
    }
}

/// Value of row `l`, column `j` of packed int4 weights (see [`Backend::dequant_int4`]);
/// `q` holds the bytes as integers.
pub fn int4_value<T: Copy + Into<i32>>(q: &[T], l: usize, j: usize, n: usize) -> f32 {
    let b = (q[(l / 2) * n + j].into() & 0xFF) as u8;
    let v = if l % 2 == 1 { b >> 4 } else { b & 0xF } as i32;
    (if v >= 8 { v - 16 } else { v }) as f32
}

/// Attention by blocks of query rows (`matmul` + `softmax` per block), with
/// at most `max_scores` score elements per block; the default of
/// [`Backend::attention`] (with 16M). Exact for any block size.
pub fn attention_chunked<B: Backend + ?Sized>(
    be: &B,
    q: &Storage,
    k: &Storage,
    v: &Storage,
    mask: Option<&Storage>,
    spec: AttentionSpec,
    max_scores: usize,
) -> Storage {
    let AttentionSpec { b, h, tq, tk, d, dv, scale, mask_shape } = spec;
    let bh = b * h;
    let block = |i0: usize, n: usize| -> Storage {
        let qb = if n == tq { q.clone() } else { be.narrow(q, &[b, h, tq, d], 2, i0, n) };
        let qk = MatmulSpec { batch: bh, m: n, k: d, n: tk, trans_a: false, trans_b: true, a_batch_stride: n * d, b_batch_stride: tk * d };
        let shape = [b, h, n, tk];
        let mut s = be.cast(&be.matmul(&qb, k, qk), DType::F32);
        s = be.binary(BinaryOp::Mul, &s, &shape, &be.from_f32(vec![scale]), &[], &shape);
        if let (Some(m), Some(ms)) = (mask, mask_shape) {
            let (m, ms) = if ms[2] == 1 || n == tq {
                (m.clone(), ms)
            } else {
                (be.narrow(m, &ms, 2, i0, n), [ms[0], ms[1], n, ms[3]])
            };
            s = be.binary(BinaryOp::Add, &s, &shape, &m, &ms, &shape);
        }
        let p = be.cast(&be.softmax(&s, &shape, 3), q.dtype());
        let pv = MatmulSpec { batch: bh, m: n, k: tk, n: dv, trans_a: false, trans_b: false, a_batch_stride: n * tk, b_batch_stride: tk * dv };
        be.matmul(&p, v, pv)
    };
    let rows = (max_scores / (bh * tk).max(1)).clamp(1, tq.max(1));
    if rows >= tq {
        return block(0, tq);
    }
    let parts: Vec<(Storage, [usize; 4])> =
        (0..tq).step_by(rows).map(|i0| (block(i0, rows.min(tq - i0)), [b, h, rows.min(tq - i0), dv])).collect();
    let refs: Vec<&Storage> = parts.iter().map(|p| &p.0).collect();
    let shapes: Vec<&[usize]> = parts.iter().map(|p| &p.1[..]).collect();
    be.cat(&refs, &shapes, 2)
}

/// Source element index of every output element of a strided view, in
/// row-major output order (see [`Backend::strided_copy`]).
pub fn strided_indices(shape: &[usize], strides: &[usize], offset: usize) -> Vec<usize> {
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    let mut pos = offset;
    for _ in 0..n {
        out.push(pos);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            pos += strides[d];
            if idx[d] < shape[d] {
                break;
            }
            pos -= strides[d] * shape[d];
            idx[d] = 0;
        }
    }
    out
}

/// Host-memory buffer of one dtype.
#[derive(Clone)]
pub enum CpuStorage {
    /// `f32` elements.
    F32(Arc<Vec<f32>>),
    /// `f16` elements.
    F16(Arc<Vec<half::f16>>),
    /// `bf16` elements.
    BF16(Arc<Vec<half::bf16>>),
    /// `i32` elements.
    I32(Arc<Vec<i32>>),
    /// `i8` elements.
    I8(Arc<Vec<i8>>),
}

/// Device-resident buffer. Cheap to clone (reference counted); storages are
/// never mutated in place once created.
#[derive(Clone)]
pub enum Storage {
    /// Host memory.
    Cpu(CpuStorage),
    /// NVIDIA GPU memory.
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaStorage),
    /// Apple Metal buffer.
    #[cfg(feature = "mps")]
    Mps(mps::MpsStorage),
}

impl Storage {
    /// Device the buffer lives on.
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => Device::Cuda(s.ordinal),
            #[cfg(feature = "mps")]
            Storage::Mps(_) => Device::Mps,
        }
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(CpuStorage::F32(_)) => DType::F32,
            Storage::Cpu(CpuStorage::F16(_)) => DType::F16,
            Storage::Cpu(CpuStorage::BF16(_)) => DType::BF16,
            Storage::Cpu(CpuStorage::I32(_)) => DType::I32,
            Storage::Cpu(CpuStorage::I8(_)) => DType::I8,
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => s.dtype,
            #[cfg(feature = "mps")]
            Storage::Mps(s) => s.dtype,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(CpuStorage::F32(v)) => v.len(),
            Storage::Cpu(CpuStorage::F16(v)) => v.len(),
            Storage::Cpu(CpuStorage::BF16(v)) => v.len(),
            Storage::Cpu(CpuStorage::I32(v)) => v.len(),
            Storage::Cpu(CpuStorage::I8(v)) => v.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => s.len,
            #[cfg(feature = "mps")]
            Storage::Mps(s) => s.len,
        }
    }

    /// `true` if the storage has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Backend of this storage's device.
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
            DType::I8 => be.cast(&be.from_i32(self.to_vec_i32()), DType::I8),
            // through f32 on the host (exact: every half value is an f32)
            d => be.cast(&be.from_f32(self.to_vec()), d),
        }
    }
}

/// Elementwise unary operation (see [`Backend::unary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `e^x`
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// `1 / x`
    Recip,
    /// `|x|`
    Abs,
    /// Hyperbolic tangent.
    Tanh,
    /// `1 / (1 + e^-x)`
    Sigmoid,
    /// `max(x, 0)`
    Relu,
    /// GELU (tanh approximation)
    Gelu,
    /// d/dx GELU (tanh approximation)
    GeluGrad,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
}

/// Elementwise binary operation with broadcasting (see [`Backend::binary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `a ^ b`
    Pow,
    /// Elementwise `max(a, b)`.
    Maximum,
    /// Elementwise `min(a, b)`.
    Minimum,
    /// 1.0 where equal, else 0.0
    Eq,
    /// 1.0 where a > b, else 0.0
    Gt,
    /// 1.0 where a < b, else 0.0
    Lt,
}

/// Reduction along one axis (see [`Backend::reduce`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
    /// Index of the maximum along the axis (I32 output).
    ArgMax,
}

/// Parameters for im2col / col2im (2D convolution lowering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dGeom {
    /// Batch size.
    pub n: usize,
    /// Input channels.
    pub c: usize,
    /// Input height.
    pub h: usize,
    /// Input width.
    pub w: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Stride (both axes).
    pub stride: usize,
    /// Zero padding (both axes).
    pub padding: usize,
}

impl Conv2dGeom {
    /// Output spatial size `(OH, OW)`.
    pub fn out_hw(&self) -> (usize, usize) {
        let oh = (self.h + 2 * self.padding - self.kh) / self.stride + 1;
        let ow = (self.w + 2 * self.padding - self.kw) / self.stride + 1;
        (oh, ow)
    }
}

/// Geometry of a fused attention call ([`Backend::attention`]).
#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    /// Batch size.
    pub b: usize,
    /// Heads.
    pub h: usize,
    /// Query positions.
    pub tq: usize,
    /// Key / value positions.
    pub tk: usize,
    /// Query / key size.
    pub d: usize,
    /// Value size.
    pub dv: usize,
    /// Multiplier of `q · k` (usually `1 / sqrt(d)`).
    pub scale: f32,
    /// Shape of the additive `F32` mask, broadcastable to `[b, h, tq, tk]`
    /// (each dim 1 or full, the last always `tk`); `None` = no mask.
    pub mask_shape: Option<[usize; 4]>,
}

impl AttentionSpec {
    /// Element strides of the mask for (batch, head, query); 0 where broadcast.
    pub fn mask_strides(&self) -> [usize; 3] {
        let Some(ms) = self.mask_shape else { return [0; 3] };
        [
            if ms[0] == 1 { 0 } else { ms[1] * ms[2] * self.tk },
            if ms[1] == 1 { 0 } else { ms[2] * self.tk },
            if ms[2] == 1 { 0 } else { self.tk },
        ]
    }
}

/// Batched GEMM description: for each batch `i`,
/// `C[i] (m x n) = op(A[i]) (m x k) @ op(B[i]) (k x n)`,
/// where `op` optionally transposes. `A[i]` starts at `i * a_batch_stride`
/// (0 = broadcast the same matrix), likewise for `B`. `C` is contiguous.
#[derive(Clone, Copy, Debug)]
pub struct MatmulSpec {
    /// Number of matrix products.
    pub batch: usize,
    /// Rows of `op(A)` and `C`.
    pub m: usize,
    /// Inner (contracted) dimension.
    pub k: usize,
    /// Columns of `op(B)` and `C`.
    pub n: usize,
    /// Transpose `A[i]` (stored as `k x m`).
    pub trans_a: bool,
    /// Transpose `B[i]` (stored as `n x k`).
    pub trans_b: bool,
    /// Element offset between consecutive `A[i]` (0 = broadcast).
    pub a_batch_stride: usize,
    /// Element offset between consecutive `B[i]` (0 = broadcast).
    pub b_batch_stride: usize,
}

/// Hyper-parameters of one Adam/AdamW update (bias corrections precomputed).
#[derive(Clone, Copy, Debug)]
pub struct AdamParams {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Added to the denominator for numerical stability.
    pub eps: f32,
    /// Weight-decay coefficient (0 = none).
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
/// Float inputs may be `F32`, `F16` or `BF16` (all operands of one call share
/// a dtype; tensor-level ops promote before calling) and outputs keep that
/// dtype, except where noted. Half inputs are computed with f32 internally.
/// Index arguments are `I32` storages on the same device. `adam_step`,
/// `index_add` and `scatter_add` are only required for `F32`. Each method maps closely to a standard
/// GPU kernel / library call (cuBLAS / MPSMatrixMultiplication for `matmul`,
/// simple elementwise kernels for `unary`/`binary`, etc.).
#[allow(clippy::wrong_self_convention)]
pub trait Backend: Send + Sync {
    /// Device this backend runs on.
    fn device(&self) -> Device;

    /// Block until all queued work on this device has finished
    /// (no-op for synchronous backends such as the CPU).
    fn synchronize(&self) {}

    // ---- memory ----
    /// Upload `f32` data as an `F32` storage.
    fn from_f32(&self, data: Vec<f32>) -> Storage;
    /// Upload `i32` data as an `I32` storage.
    fn from_i32(&self, data: Vec<i32>) -> Storage;
    /// Copy to host as f32 (any dtype).
    fn to_f32(&self, x: &Storage) -> Vec<f32>;
    /// Copy to host as i32 (any dtype).
    fn to_i32(&self, x: &Storage) -> Vec<i32>;
    /// `len` elements of `dtype`, all set to `value`.
    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage;
    /// Upload int8 data (default: through i32 and a cast).
    fn from_i8(&self, data: Vec<i8>) -> Storage {
        self.cast(&self.from_i32(data.into_iter().map(i32::from).collect()), DType::I8)
    }
    /// Convert to `dtype` (returns `x` itself if it already has that dtype).
    fn cast(&self, x: &Storage, dtype: DType) -> Storage;

    // ---- elementwise ----
    /// Elementwise unary op (float dtypes).
    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage;
    /// Elementwise binary op with NumPy broadcasting into `out_shape`.
    /// Comparison ops (`Eq`/`Gt`/`Lt`) return a 0/1 mask in the inputs' float
    /// dtype, and also accept `I32` inputs (then the mask is `F32`).
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
    /// Batched GEMM described by `spec` (see [`MatmulSpec`]).
    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage;
    /// Generic axis permutation (copy).
    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage;
    /// Contiguous copy of a strided view: output element at multi-index `i`
    /// is `x[offset + Σ i[d] * strides[d]]` (strides in elements, 0 for
    /// broadcast dims; any dtype). Used to materialize tensor views. The
    /// default gathers on the host; backends override it.
    fn strided_copy(&self, x: &Storage, shape: &[usize], strides: &[usize], offset: usize) -> Storage {
        let idx = strided_indices(shape, strides, offset);
        match x.dtype() {
            DType::I32 | DType::I8 => {
                let v = x.to_vec_i32();
                self.cast(&self.from_i32(idx.iter().map(|&i| v[i]).collect()), x.dtype())
            }
            d => {
                let v = x.to_vec();
                self.cast(&self.from_f32(idx.iter().map(|&i| v[i]).collect()), d)
            }
        }
    }
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

    /// Weight-only int8 matmul: `x (m x k) @ (q (k x n, I8) * scale[n])`,
    /// output in `x`'s dtype. `scales` is `F32`. The default dequantizes and
    /// calls `matmul`; backends override it to read the int8 weights directly
    /// (a quarter of the memory traffic, which dominates token-by-token decoding).
    fn matmul_int8(&self, x: &Storage, q: &Storage, scales: &Storage, m: usize, k: usize, n: usize) -> Storage {
        let dt = x.dtype();
        let w = self.cast(q, dt);
        let sc = self.cast(scales, dt);
        let w = self.binary(BinaryOp::Mul, &w, &[k, n], &sc, &[n], &[k, n]);
        let spec = MatmulSpec { batch: 1, m, k, n, trans_a: false, trans_b: false, a_batch_stride: 0, b_batch_stride: 0 };
        self.matmul(x, &w, spec)
    }

    /// Dequantize group-wise int4 weights to an `F32` `[k, n]` matrix.
    /// `q` (`I8`, `k/2 * n` bytes) packs rows `2r` (low nibble) and `2r+1`
    /// (high nibble) of column `j` into byte `r * n + j`, as 4-bit two's
    /// complement; `scales` (`F32`, `[k / group, n]`) has one scale per
    /// `group` rows. The default unpacks on the host; backends override it.
    fn dequant_int4(&self, q: &Storage, scales: &Storage, k: usize, n: usize, group: usize) -> Storage {
        let (q, sc) = (q.to_vec_i32(), scales.to_vec());
        let w: Vec<f32> = (0..k * n).map(|t| int4_value(&q, t / n, t % n, n) * sc[(t / n / group) * n + t % n]).collect();
        self.from_f32(w)
    }

    /// Weight-only int4 matmul `x (m x k) @ dequant_int4(q, scales)`, output in
    /// `x`'s dtype (layout as in [`Backend::dequant_int4`]). The default
    /// dequantizes and calls `matmul`; backends override it to read the
    /// packed weights directly (1/8 of the f32 weight traffic).
    #[allow(clippy::too_many_arguments)]
    fn matmul_int4(&self, x: &Storage, q: &Storage, scales: &Storage, m: usize, k: usize, n: usize, group: usize) -> Storage {
        let w = self.cast(&self.dequant_int4(q, scales, k, n, group), x.dtype());
        let spec = MatmulSpec { batch: 1, m, k, n, trans_a: false, trans_b: false, a_batch_stride: 0, b_batch_stride: 0 };
        self.matmul(x, &w, spec)
    }

    /// Scaled dot-product attention `softmax(q kᵀ * scale + mask) v` for
    /// contiguous `q` `[b, h, tq, d]`, `k` `[b, h, tk, d]`, `v` `[b, h, tk, dv]`
    /// (all one float dtype) and an optional `F32` `mask` of shape
    /// `spec.mask_shape`; returns `[b, h, tq, dv]` in `q`'s dtype.
    ///
    /// The default processes blocks of query rows with `matmul` + `softmax`,
    /// sized so a block's scores stay under 16M elements (64 MB in f32):
    /// exact, GEMM speed, and bounded memory for long sequences. Backends may
    /// override it with a fused kernel (the CPU uses FlashAttention-style
    /// tiling with an online softmax).
    fn attention(&self, q: &Storage, k: &Storage, v: &Storage, mask: Option<&Storage>, spec: AttentionSpec) -> Storage {
        attention_chunked(self, q, k, v, mask, spec, 1 << 24)
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
        fn strided_copy(&self, x: &Storage, s: &[usize], st: &[usize], o: usize) -> Storage {
            self.0.strided_copy(x, s, st, o)
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
