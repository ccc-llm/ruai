//! Apple Metal (MPS) backend — skeleton (enabled with `--features mps`).
//!
//! Plan for the real implementation (e.g. with the `metal` / `objc2-metal` crates):
//!
//! | primitive        | implementation                                          |
//! |------------------|---------------------------------------------------------|
//! | storage          | `MTLBuffer` with `StorageModeShared` (unified memory: `to_vec` is a memcpy) |
//! | `unary`/`binary`/`expand` | Metal compute kernels (one `.metal` library) |
//! | `reduce`         | threadgroup reduction kernel                            |
//! | `matmul`         | `MPSMatrixMultiplication` (supports transposes + batch) |
//! | `permute`, `index_*`, `im2col/col2im` | Metal compute kernels              |
//!
//! Ops would be encoded into a shared command buffer and committed lazily
//! (flushed on `to_vec`) to avoid per-op synchronization.

use super::{Backend, BinaryOp, Conv2dGeom, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::device::Device;

/// A real implementation would hold e.g. `Arc<metal::Buffer>`.
#[derive(Clone)]
pub struct MpsStorage {
    pub len: usize,
}

pub struct MpsBackend;

static MPS: MpsBackend = MpsBackend;

pub fn backend() -> &'static dyn Backend {
    &MPS
}

macro_rules! not_yet {
    ($what:expr) => {
        unimplemented!("MPS backend: `{}` is not implemented yet", $what)
    };
}

impl Backend for MpsBackend {
    fn device(&self) -> Device {
        Device::Mps
    }
    fn from_vec(&self, _data: Vec<f32>) -> Storage {
        not_yet!("from_vec")
    }
    fn to_vec(&self, _x: &Storage) -> Vec<f32> {
        not_yet!("to_vec")
    }
    fn full(&self, _len: usize, _value: f32) -> Storage {
        not_yet!("full")
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
    fn reduce(&self, _op: ReduceOp, _x: &Storage, _s: &[usize], _axis: usize) -> Storage {
        not_yet!("reduce")
    }
    fn matmul(&self, _a: &Storage, _b: &Storage, _spec: MatmulSpec) -> Storage {
        not_yet!("matmul")
    }
    fn permute(&self, _x: &Storage, _s: &[usize], _perm: &[usize]) -> Storage {
        not_yet!("permute")
    }
    fn index_select(&self, _t: &Storage, _cols: usize, _idx: &[usize]) -> Storage {
        not_yet!("index_select")
    }
    fn index_add(&self, _s: &Storage, _rows: usize, _cols: usize, _idx: &[usize]) -> Storage {
        not_yet!("index_add")
    }
    fn im2col(&self, _x: &Storage, _g: Conv2dGeom) -> Storage {
        not_yet!("im2col")
    }
    fn col2im(&self, _c: &Storage, _g: Conv2dGeom) -> Storage {
        not_yet!("col2im")
    }
}
