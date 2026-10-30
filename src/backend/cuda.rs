//! CUDA backend — skeleton (enabled with `--features cuda`).
//!
//! Plan for the real implementation (e.g. with the `cudarc` crate):
//!
//! | primitive        | implementation                                         |
//! |------------------|--------------------------------------------------------|
//! | `from_*/to_*`    | `cuMemcpyHtoD` / `cuMemcpyDtoH`                         |
//! | `full`           | fill kernel / `cuMemsetD32`                            |
//! | `unary`          | one elementwise kernel per `UnaryOp` (NVRTC-compiled)  |
//! | `binary`/`expand`| broadcast kernel taking shape + strides as arguments   |
//! | `reduce`         | (outer, len, inner) reduction kernel, warp shuffles    |
//! | `matmul`         | `cublasSgemmStridedBatched` (trans flags + strides map 1:1) |
//! | `permute`        | strided-copy kernel                                    |
//! | `index_select/add` | gather / `atomicAdd` scatter kernels                |
//! | `im2col/col2im`  | standard Caffe-style kernels (or cuDNN conv directly)  |
//!
//! Only this file (and `CudaStorage`) has to change; autograd, `nn` and
//! `optim` are written purely against the [`Backend`] trait.

use super::{Backend, BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::device::Device;

/// Device buffer handle. A real implementation would hold e.g.
/// `Arc<cudarc::driver::CudaSlice<f32>>`.
#[derive(Clone)]
pub struct CudaStorage {
    pub ordinal: usize,
    pub len: usize,
    pub dtype: DType,
}

pub struct CudaBackend {
    pub ordinal: usize,
}

static CUDA0: CudaBackend = CudaBackend { ordinal: 0 };

pub fn backend(ordinal: usize) -> &'static dyn Backend {
    assert_eq!(ordinal, 0, "only cuda:0 is wired up in the skeleton");
    &CUDA0
}

macro_rules! not_yet {
    ($what:expr) => {
        unimplemented!("CUDA backend: `{}` is not implemented yet", $what)
    };
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }
    fn from_f32(&self, _data: Vec<f32>) -> Storage {
        not_yet!("from_f32")
    }
    fn from_i32(&self, _data: Vec<i32>) -> Storage {
        not_yet!("from_i32")
    }
    fn to_f32(&self, _x: &Storage) -> Vec<f32> {
        not_yet!("to_f32")
    }
    fn to_i32(&self, _x: &Storage) -> Vec<i32> {
        not_yet!("to_i32")
    }
    fn full(&self, _len: usize, _value: f32, _dtype: DType) -> Storage {
        not_yet!("full")
    }
    fn cast(&self, _x: &Storage, _dtype: DType) -> Storage {
        not_yet!("cast")
    }
    fn unary(&self, _op: UnaryOp, _x: &Storage) -> Storage {
        not_yet!("unary")
    }
    fn binary(&self, _op: BinaryOp, _a: &Storage, _as: &[usize], _b: &Storage, _bs: &[usize], _os: &[usize]) -> Storage {
        not_yet!("binary")
    }
    fn expand(&self, _x: &Storage, _s: &[usize], _os: &[usize]) -> Storage {
        not_yet!("expand")
    }
    fn where_cond(&self, _c: &Storage, _a: &Storage, _b: &Storage) -> Storage {
        not_yet!("where_cond")
    }
    fn reduce(&self, _op: ReduceOp, _x: &Storage, _s: &[usize], _axis: usize) -> Storage {
        not_yet!("reduce")
    }
    fn matmul(&self, _a: &Storage, _b: &Storage, _spec: MatmulSpec) -> Storage {
        not_yet!("matmul")
    }
    fn permute(&self, _x: &Storage, _s: &[usize], _perm: &[usize]) -> Storage {
        not_yet!("permute")
    }
    fn narrow(&self, _x: &Storage, _s: &[usize], _axis: usize, _start: usize, _len: usize) -> Storage {
        not_yet!("narrow")
    }
    fn cat(&self, _xs: &[&Storage], _shapes: &[&[usize]], _axis: usize) -> Storage {
        not_yet!("cat")
    }
    fn index_select(&self, _t: &Storage, _cols: usize, _idx: &Storage) -> Storage {
        not_yet!("index_select")
    }
    fn index_add(&self, _s: &Storage, _rows: usize, _cols: usize, _idx: &Storage) -> Storage {
        not_yet!("index_add")
    }
    fn gather(&self, _x: &Storage, _s: &[usize], _axis: usize, _idx: &Storage, _l: usize) -> Storage {
        not_yet!("gather")
    }
    fn scatter_add(&self, _src: &Storage, _idx: &Storage, _l: usize, _s: &[usize], _axis: usize) -> Storage {
        not_yet!("scatter_add")
    }
    fn im2col(&self, _x: &Storage, _g: Conv2dGeom) -> Storage {
        not_yet!("im2col")
    }
    fn col2im(&self, _c: &Storage, _g: Conv2dGeom) -> Storage {
        not_yet!("col2im")
    }
}
