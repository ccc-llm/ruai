//! Backend abstraction.
//!
//! Every tensor operation in the engine (including autograd and all `nn` layers)
//! is built from the small set of *primitives* in the [`Backend`] trait. All
//! storages are contiguous, row-major `f32` buffers.
//!
//! To add a new device (CUDA, MPS, ...) you only need to:
//!   1. add a variant to [`Storage`],
//!   2. implement [`Backend`] for it,
//!   3. return it from [`backend_for`].
//!
//! Nothing above this module needs to change.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "mps")]
pub mod mps;

use crate::device::Device;
use std::sync::Arc;

/// Device-resident buffer of `f32`s. Cheap to clone (reference counted);
/// storages are never mutated in place once created.
#[derive(Clone)]
pub enum Storage {
    Cpu(Arc<Vec<f32>>),
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

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
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

    /// Copy the data to host memory.
    pub fn to_vec(&self) -> Vec<f32> {
        self.backend().to_vec(self)
    }

    /// Move/copy the storage to another device (through host memory).
    pub fn to_device(&self, device: Device) -> Storage {
        if self.device() == device {
            return self.clone();
        }
        backend_for(device).from_vec(self.to_vec())
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
    /// Index (as f32) of the maximum along the axis.
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

/// The primitive operations a device must provide.
///
/// Shapes are passed explicitly; all inputs and outputs are contiguous.
/// Each method maps closely to a standard GPU kernel / library call
/// (cuBLAS / MPSMatrixMultiplication for `matmul`, simple elementwise kernels
/// for `unary`/`binary`, etc.).
#[allow(clippy::wrong_self_convention)]
pub trait Backend: Send + Sync {
    fn device(&self) -> Device;

    // ---- memory ----
    fn from_vec(&self, data: Vec<f32>) -> Storage;
    fn to_vec(&self, x: &Storage) -> Vec<f32>;
    fn full(&self, len: usize, value: f32) -> Storage;

    // ---- elementwise ----
    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage;
    /// Elementwise binary op with NumPy broadcasting into `out_shape`.
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

    // ---- reductions ----
    /// Reduce over `axis` of `shape`; output has that axis removed.
    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage;

    // ---- linear algebra / layout ----
    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage;
    /// Generic axis permutation (copy).
    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage;

    // ---- indexing ----
    /// Gather rows: `table` is `[rows, cols]`, output `[idx.len(), cols]`.
    fn index_select(&self, table: &Storage, cols: usize, idx: &[usize]) -> Storage;
    /// Scatter-add rows of `src` (`[idx.len(), cols]`) into a zero `[rows, cols]`.
    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &[usize]) -> Storage;

    // ---- convolution lowering ----
    /// `[N, C, H, W]` -> `[N, C*KH*KW, OH*OW]`
    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage;
    /// Adjoint of `im2col`: `[N, C*KH*KW, OH*OW]` -> `[N, C, H, W]` (accumulating).
    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage;
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
