//! NVIDIA CUDA backend (`--features cuda`), built on `cudarc`.
//!
//! * The CUDA driver, NVRTC and cuBLAS are loaded at runtime, so the crate
//!   builds anywhere; `Device::Cuda(i).is_available()` reports whether a GPU
//!   and driver are present.
//! * Kernels: `cuda_kernels.cu` (a port of `mps_kernels.metal`), compiled with
//!   NVRTC on first use with precise math.
//! * Matmul: `cublasSgemmStridedBatched`, using the row-major trick
//!   `C = A·B  <=>  Cᵀ = Bᵀ·Aᵀ` in cuBLAS's column-major convention.
//! * Everything runs asynchronously on one stream per device; memory is
//!   stream-ordered (`cuMemAllocAsync` / `cuMemFreeAsync`), so dropping a
//!   storage while kernels still use it is safe. `to_f32`/`to_i32` synchronize.

use super::{AdamParams, Backend, BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_strides, numel, split_axis, strides};
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const KERNELS: &str = include_str!("cuda_kernels.cu");
const KERNEL_NAMES: &[&str] = &[
    "fill_u32",
    "cast_f32_i32",
    "cast_i32_f32",
    "unary_f32",
    "binary_f32",
    "strided_copy",
    "cat_copy",
    "where_f32",
    "reduce_strided",
    "reduce_rows",
    "softmax_rows",
    "softmax_backward_rows",
    "layer_norm_rows",
    "layer_norm_backward_rows",
    "adam_step",
    "index_select",
    "index_add",
    "gather",
    "scatter_add",
    "im2col",
    "col2im",
];
const MAX_RANK: usize = 8;
const BLOCK: u32 = 256;

#[derive(Clone)]
enum Buf {
    F32(Arc<CudaSlice<f32>>),
    I32(Arc<CudaSlice<i32>>),
}

/// Device buffer on one GPU.
#[derive(Clone)]
pub struct CudaStorage {
    buf: Buf,
    pub ordinal: usize,
    pub len: usize,
    pub dtype: DType,
}

pub struct CudaBackend {
    ordinal: usize,
    stream: Arc<CudaStream>,
    blas: Mutex<CudaBlas>,
    funcs: HashMap<&'static str, CudaFunction>,
}

static BACKENDS: OnceLock<Mutex<HashMap<usize, &'static CudaBackend>>> = OnceLock::new();

/// Whether the CUDA driver is installed and GPU `ordinal` exists.
pub fn is_available(ordinal: usize) -> bool {
    // Safety: only probes whether the driver library can be loaded.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return false;
    }
    CudaContext::device_count().map(|n| (ordinal as i32) < n).unwrap_or(false)
}

pub fn backend(ordinal: usize) -> &'static dyn Backend {
    let map = BACKENDS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    *map.entry(ordinal).or_insert_with(|| Box::leak(Box::new(CudaBackend::new(ordinal))))
}

// ---- kernel parameter structs (mirror cuda_kernels.cu) ----

#[repr(C)]
#[derive(Clone, Copy)]
struct BinParams {
    n: u32,
    rank: u32,
    op: u32,
    pad: u32,
    shape: [u32; MAX_RANK],
    sa: [u32; MAX_RANK],
    sb: [u32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyParams {
    n: u32,
    rank: u32,
    base: u32,
    pad: u32,
    shape: [u32; MAX_RANK],
    strides: [u32; MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CatParams {
    n: u32,
    blk_in: u32,
    blk_out: u32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RedParams {
    outer: u32,
    len: u32,
    inner: u32,
    op: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdamKernelParams {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    decoupled: u32,
    ib1: f32,
    ib2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GatherParams {
    n: u32,
    len: u32,
    idx_len: u32,
    inner: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    padding: u32,
    oh: u32,
    ow: u32,
}

// Safety: plain #[repr(C)] structs of 32-bit fields, passed by value to kernels.
unsafe impl DeviceRepr for BinParams {}
unsafe impl DeviceRepr for CopyParams {}
unsafe impl DeviceRepr for CatParams {}
unsafe impl DeviceRepr for RedParams {}
unsafe impl DeviceRepr for AdamKernelParams {}
unsafe impl DeviceRepr for GatherParams {}
unsafe impl DeviceRepr for ConvParams {}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("CUDA backend: size exceeds u32")
}

fn fill8(v: &[usize]) -> [u32; MAX_RANK] {
    assert!(v.len() <= MAX_RANK, "CUDA backend supports tensors of rank <= {MAX_RANK}");
    let mut a = [0u32; MAX_RANK];
    for (d, &x) in a.iter_mut().zip(v) {
        *d = u(x);
    }
    a
}

fn st(s: &Storage) -> &CudaStorage {
    match s {
        Storage::Cuda(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("CUDA backend received a non-CUDA storage ({})", s.device()),
    }
}

fn wrap(s: CudaStorage) -> Storage {
    Storage::Cuda(s)
}

/// One thread per element.
fn threads(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (u(n).div_ceil(BLOCK).max(1), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
}

/// One block per row; block size a multiple of 32 (<= 256).
fn rows(rows: usize, d: usize) -> LaunchConfig {
    let tb = (d.div_ceil(32).clamp(1, 8) * 32) as u32;
    LaunchConfig { grid_dim: (u(rows), 1, 1), block_dim: (tb, 1, 1), shared_mem_bytes: 0 }
}

fn unary_code(op: UnaryOp) -> u32 {
    match op {
        UnaryOp::Neg => 0,
        UnaryOp::Exp => 1,
        UnaryOp::Log => 2,
        UnaryOp::Sqrt => 3,
        UnaryOp::Recip => 4,
        UnaryOp::Abs => 5,
        UnaryOp::Tanh => 6,
        UnaryOp::Sigmoid => 7,
        UnaryOp::Relu => 8,
        UnaryOp::Gelu => 9,
        UnaryOp::GeluGrad => 10,
        UnaryOp::Sin => 11,
        UnaryOp::Cos => 12,
    }
}

fn binary_code(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Pow => 4,
        BinaryOp::Maximum => 5,
        BinaryOp::Minimum => 6,
        BinaryOp::Eq => 7,
        BinaryOp::Gt => 8,
        BinaryOp::Lt => 9,
    }
}

/// Launch kernel `$name` with `$cfg`; each argument is a reference to a
/// device pointer (`u64`), a scalar, or a parameter struct.
macro_rules! launch {
    ($be:expr, $name:literal, $cfg:expr; $($arg:expr),* $(,)?) => {{
        let cfg: LaunchConfig = $cfg;
        if cfg.grid_dim.0 > 0 {
            let mut b = $be.stream.launch_builder($be.func($name));
            $( b.arg($arg); )*
            unsafe { b.launch(cfg) }.unwrap_or_else(|e| panic!("CUDA kernel `{}` failed: {e:?}", $name));
        }
    }};
}

impl CudaBackend {
    fn new(ordinal: usize) -> Self {
        assert!(is_available(ordinal), "CUDA device {ordinal} is not available (no driver or no such GPU)");
        let ctx = CudaContext::new(ordinal).unwrap_or_else(|e| panic!("cannot open CUDA device {ordinal}: {e:?}"));
        // One stream per device, so cross-stream event tracking is unnecessary;
        // this also lets kernels take raw device pointers.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let opts = CompileOptions { use_fast_math: Some(false), ..Default::default() };
        let ptx = compile_ptx_with_opts(KERNELS, opts).unwrap_or_else(|e| panic!("NVRTC failed to compile kernels: {e:?}"));
        let module = ctx.load_module(ptx).unwrap_or_else(|e| panic!("failed to load CUDA module: {e:?}"));
        let funcs = KERNEL_NAMES
            .iter()
            .map(|&n| (n, module.load_function(n).unwrap_or_else(|e| panic!("CUDA kernel `{n}` not found: {e:?}"))))
            .collect();
        let blas = CudaBlas::new(stream.clone()).unwrap_or_else(|e| panic!("cuBLAS init failed: {e:?}"));
        CudaBackend { ordinal, stream, blas: Mutex::new(blas), funcs }
    }

    fn func(&self, name: &str) -> &CudaFunction {
        &self.funcs[name]
    }

    /// Uninitialized storage (stream-ordered allocation).
    fn alloc(&self, len: usize, dtype: DType) -> CudaStorage {
        let n = len.max(1);
        let buf = unsafe {
            match dtype {
                DType::F32 => Buf::F32(Arc::new(self.stream.alloc::<f32>(n).expect("CUDA alloc failed"))),
                DType::I32 => Buf::I32(Arc::new(self.stream.alloc::<i32>(n).expect("CUDA alloc failed"))),
            }
        };
        CudaStorage { buf, ordinal: self.ordinal, len, dtype }
    }

    fn ptr(&self, s: &CudaStorage) -> u64 {
        match &s.buf {
            Buf::F32(b) => b.device_ptr(&self.stream).0,
            Buf::I32(b) => b.device_ptr(&self.stream).0,
        }
    }

    fn to_f32_storage(&self, s: &CudaStorage) -> CudaStorage {
        if s.dtype == DType::F32 {
            return s.clone();
        }
        let out = self.alloc(s.len, DType::F32);
        let (x, y, n) = (self.ptr(s), self.ptr(&out), u(s.len));
        launch!(self, "cast_i32_f32", threads(s.len); &x, &y, &n);
        out
    }

    fn zeros(&self, len: usize) -> CudaStorage {
        let out = self.alloc(len, DType::F32);
        let (o, n, bits) = (self.ptr(&out), u(len), 0u32);
        launch!(self, "fill_u32", threads(len); &o, &n, &bits);
        out
    }

    /// Generic strided gather copy (any 32-bit dtype).
    fn strided_copy(&self, x: &CudaStorage, out_shape: &[usize], in_strides: &[usize], base: usize) -> CudaStorage {
        let n = numel(out_shape);
        let out = self.alloc(n, x.dtype);
        let p = CopyParams {
            n: u(n),
            rank: u(out_shape.len()),
            base: u(base),
            pad: 0,
            shape: fill8(out_shape),
            strides: fill8(in_strides),
        };
        let (xp, op) = (self.ptr(x), self.ptr(&out));
        launch!(self, "strided_copy", threads(n); &xp, &op, &p);
        out
    }

    fn f32_slice<'a>(&self, s: &'a CudaStorage) -> &'a CudaSlice<f32> {
        match &s.buf {
            Buf::F32(b) => b,
            Buf::I32(_) => panic!("CUDA matmul expects f32 inputs"),
        }
    }

    fn softmax_rows(&self, x: &Storage, shape: &[usize], log: bool) -> Storage {
        let d = shape[shape.len() - 1];
        let r = numel(shape) / d.max(1);
        let out = self.alloc(r * d, DType::F32);
        let (xp, yp, du, lm) = (self.ptr(st(x)), self.ptr(&out), u(d), log as u32);
        launch!(self, "softmax_rows", rows(r, d); &xp, &yp, &du, &lm);
        wrap(out)
    }

    fn softmax_backward_rows(&self, y: &Storage, g: &Storage, shape: &[usize], log: bool) -> Storage {
        let d = shape[shape.len() - 1];
        let r = numel(shape) / d.max(1);
        let out = self.alloc(r * d, DType::F32);
        let (yp, gp, op, du, lm) = (self.ptr(st(y)), self.ptr(st(g)), self.ptr(&out), u(d), log as u32);
        launch!(self, "softmax_backward_rows", rows(r, d); &yp, &gp, &op, &du, &lm);
        wrap(out)
    }
}

fn conv_params(g: Conv2dGeom) -> ConvParams {
    let (oh, ow) = g.out_hw();
    ConvParams {
        n: u(g.n),
        c: u(g.c),
        h: u(g.h),
        w: u(g.w),
        kh: u(g.kh),
        kw: u(g.kw),
        stride: u(g.stride),
        padding: u(g.padding),
        oh: u(oh),
        ow: u(ow),
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    fn synchronize(&self) {
        self.stream.synchronize().expect("CUDA synchronize failed");
    }

    fn from_f32(&self, data: Vec<f32>) -> Storage {
        let len = data.len();
        let host = if data.is_empty() { vec![0.0] } else { data };
        let s = self.stream.clone_htod(&host).expect("CUDA upload failed");
        wrap(CudaStorage { buf: Buf::F32(Arc::new(s)), ordinal: self.ordinal, len, dtype: DType::F32 })
    }

    fn from_i32(&self, data: Vec<i32>) -> Storage {
        let len = data.len();
        let host = if data.is_empty() { vec![0] } else { data };
        let s = self.stream.clone_htod(&host).expect("CUDA upload failed");
        wrap(CudaStorage { buf: Buf::I32(Arc::new(s)), ordinal: self.ordinal, len, dtype: DType::I32 })
    }

    fn to_f32(&self, x: &Storage) -> Vec<f32> {
        let s = st(x);
        let mut v: Vec<f32> = match &s.buf {
            Buf::F32(b) => self.stream.clone_dtoh(b.as_ref()).expect("CUDA download failed"),
            Buf::I32(b) => {
                self.stream.clone_dtoh(b.as_ref()).expect("CUDA download failed").into_iter().map(|v| v as f32).collect()
            }
        };
        v.truncate(s.len);
        v
    }

    fn to_i32(&self, x: &Storage) -> Vec<i32> {
        let s = st(x);
        let mut v: Vec<i32> = match &s.buf {
            Buf::I32(b) => self.stream.clone_dtoh(b.as_ref()).expect("CUDA download failed"),
            Buf::F32(b) => {
                self.stream.clone_dtoh(b.as_ref()).expect("CUDA download failed").into_iter().map(|v| v as i32).collect()
            }
        };
        v.truncate(s.len);
        v
    }

    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage {
        let out = self.alloc(len, dtype);
        let bits = match dtype {
            DType::F32 => value.to_bits(),
            DType::I32 => (value as i32) as u32,
        };
        let (o, n) = (self.ptr(&out), u(len));
        launch!(self, "fill_u32", threads(len); &o, &n, &bits);
        wrap(out)
    }

    fn cast(&self, x: &Storage, dtype: DType) -> Storage {
        let s = st(x);
        if s.dtype == dtype {
            return x.clone();
        }
        let out = self.alloc(s.len, dtype);
        let (xp, yp, n) = (self.ptr(s), self.ptr(&out), u(s.len));
        if dtype == DType::I32 {
            launch!(self, "cast_f32_i32", threads(s.len); &xp, &yp, &n);
        } else {
            launch!(self, "cast_i32_f32", threads(s.len); &xp, &yp, &n);
        }
        wrap(out)
    }

    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
        let s = st(x);
        assert_eq!(s.dtype, DType::F32, "CUDA unary op on {}", s.dtype);
        let out = self.alloc(s.len, DType::F32);
        let (xp, yp, n, code) = (self.ptr(s), self.ptr(&out), u(s.len), unary_code(op));
        launch!(self, "unary_f32", threads(s.len); &xp, &yp, &n, &code);
        wrap(out)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: &Storage,
        a_shape: &[usize],
        b: &Storage,
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> Storage {
        let (a, b) = (self.to_f32_storage(st(a)), self.to_f32_storage(st(b)));
        let n = numel(out_shape);
        let out = self.alloc(n, DType::F32);
        let p = BinParams {
            n: u(n),
            rank: u(out_shape.len()),
            op: binary_code(op),
            pad: 0,
            shape: fill8(out_shape),
            sa: fill8(&broadcast_strides(a_shape, out_shape)),
            sb: fill8(&broadcast_strides(b_shape, out_shape)),
        };
        let (ap, bp, op_) = (self.ptr(&a), self.ptr(&b), self.ptr(&out));
        launch!(self, "binary_f32", threads(n); &ap, &bp, &op_, &p);
        wrap(out)
    }

    fn expand(&self, x: &Storage, shape: &[usize], out_shape: &[usize]) -> Storage {
        wrap(self.strided_copy(st(x), out_shape, &broadcast_strides(shape, out_shape), 0))
    }

    fn where_cond(&self, cond: &Storage, a: &Storage, b: &Storage) -> Storage {
        let c = self.to_f32_storage(st(cond));
        let (a, b) = (st(a), st(b));
        let out = self.alloc(a.len, DType::F32);
        let (cp, ap, bp, op, n) = (self.ptr(&c), self.ptr(a), self.ptr(b), self.ptr(&out), u(a.len));
        launch!(self, "where_f32", threads(a.len); &cp, &ap, &bp, &op, &n);
        wrap(out)
    }

    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let s = st(x);
        let (outer, len, inner) = split_axis(shape, axis);
        let code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
            ReduceOp::ArgMax => 3,
        };
        let dtype = if op == ReduceOp::ArgMax { DType::I32 } else { DType::F32 };
        let out = self.alloc(outer * inner, dtype);
        let p = RedParams { outer: u(outer), len: u(len), inner: u(inner), op: code };
        let (xp, yp) = (self.ptr(s), self.ptr(&out));
        if inner == 1 && op != ReduceOp::ArgMax {
            launch!(self, "reduce_rows", rows(outer, len); &xp, &yp, &p);
        } else {
            launch!(self, "reduce_strided", threads(outer * inner); &xp, &yp, &p);
        }
        wrap(out)
    }

    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
        let MatmulSpec { batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride } = spec;
        if k == 0 || m == 0 || n == 0 {
            return wrap(self.zeros(batch * m * n));
        }
        let (a, b) = (st(a), st(b));
        let mut c = unsafe { self.stream.alloc::<f32>(batch * m * n) }.expect("CUDA alloc failed");
        // Row-major C (m x n) = op(A) op(B) is column-major Cᵀ (n x m) = op(B)ᵀ op(A)ᵀ:
        // pass B as cuBLAS's first operand and A as its second.
        let op = |t: bool| if t { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: op(trans_b),
                transb: op(trans_a),
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0f32,
                lda: if trans_b { k } else { n } as i32,
                ldb: if trans_a { m } else { k } as i32,
                beta: 0.0f32,
                ldc: n as i32,
            },
            batch_size: batch as i32,
            stride_a: if b_batch_stride == 0 { 0 } else { (k * n) as i64 },
            stride_b: if a_batch_stride == 0 { 0 } else { (m * k) as i64 },
            stride_c: (m * n) as i64,
        };
        unsafe {
            self.blas
                .lock()
                .unwrap()
                .gemm_strided_batched(cfg, self.f32_slice(b), self.f32_slice(a), &mut c)
                .unwrap_or_else(|e| panic!("cuBLAS sgemm failed: {e:?}"));
        }
        wrap(CudaStorage { buf: Buf::F32(Arc::new(c)), ordinal: self.ordinal, len: batch * m * n, dtype: DType::F32 })
    }

    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage {
        let st_in = strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let in_st: Vec<usize> = perm.iter().map(|&p| st_in[p]).collect();
        wrap(self.strided_copy(st(x), &out_shape, &in_st, 0))
    }

    fn narrow(&self, x: &Storage, shape: &[usize], axis: usize, start: usize, len: usize) -> Storage {
        let (outer, l, inner) = split_axis(shape, axis);
        wrap(self.strided_copy(st(x), &[outer, len, inner], &[l * inner, inner, 1], start * inner))
    }

    fn cat(&self, xs: &[&Storage], shapes: &[&[usize]], axis: usize) -> Storage {
        let (_, _, inner) = split_axis(shapes[0], axis);
        let total: usize = xs.iter().map(|s| s.len()).sum();
        let blk_out: usize = shapes.iter().map(|s| s[axis]).sum::<usize>() * inner;
        let out = self.alloc(total, xs[0].dtype());
        let op = self.ptr(&out);
        let mut offset = 0;
        for (x, s) in xs.iter().zip(shapes) {
            let blk_in = s[axis] * inner;
            let x = st(x);
            let p = CatParams { n: u(x.len), blk_in: u(blk_in), blk_out: u(blk_out), offset: u(offset) };
            let xp = self.ptr(x);
            launch!(self, "cat_copy", threads(x.len); &xp, &op, &p);
            offset += blk_in;
        }
        wrap(out)
    }

    fn index_select(&self, table: &Storage, cols: usize, idx: &Storage) -> Storage {
        let (t, idx) = (st(table), st(idx));
        let n = idx.len * cols;
        let out = self.alloc(n, DType::F32);
        let (tp, ip, op, nn, c) = (self.ptr(t), self.ptr(idx), self.ptr(&out), u(n), u(cols));
        launch!(self, "index_select", threads(n); &tp, &ip, &op, &nn, &c);
        wrap(out)
    }

    fn index_add(&self, src: &Storage, rows_: usize, cols: usize, idx: &Storage) -> Storage {
        let (s, idx) = (st(src), st(idx));
        let out = self.zeros(rows_ * cols);
        let (sp, ip, op, nn, c) = (self.ptr(s), self.ptr(idx), self.ptr(&out), u(s.len), u(cols));
        launch!(self, "index_add", threads(s.len); &sp, &ip, &op, &nn, &c);
        wrap(out)
    }

    fn gather(&self, x: &Storage, shape: &[usize], axis: usize, idx: &Storage, idx_len: usize) -> Storage {
        let (x, idx) = (st(x), st(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.alloc(idx.len, DType::F32);
        let p = GatherParams { n: u(idx.len), len: u(len), idx_len: u(idx_len), inner: u(inner) };
        let (xp, ip, op) = (self.ptr(x), self.ptr(idx), self.ptr(&out));
        launch!(self, "gather", threads(idx.len); &xp, &ip, &op, &p);
        wrap(out)
    }

    fn scatter_add(&self, src: &Storage, idx: &Storage, idx_len: usize, shape: &[usize], axis: usize) -> Storage {
        let (s, idx) = (st(src), st(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.zeros(numel(shape));
        let p = GatherParams { n: u(idx.len), len: u(len), idx_len: u(idx_len), inner: u(inner) };
        let (sp, ip, op) = (self.ptr(s), self.ptr(idx), self.ptr(&out));
        launch!(self, "scatter_add", threads(idx.len); &sp, &ip, &op, &p);
        wrap(out)
    }

    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
        let (oh, ow) = g.out_hw();
        let n = g.n * g.c * g.kh * g.kw * oh * ow;
        let out = self.alloc(n, DType::F32);
        let p = conv_params(g);
        let (xp, op) = (self.ptr(st(x)), self.ptr(&out));
        launch!(self, "im2col", threads(n); &xp, &op, &p);
        wrap(out)
    }

    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage {
        let n = g.n * g.c * g.h * g.w;
        let out = self.alloc(n, DType::F32);
        let p = conv_params(g);
        let (cp, op) = (self.ptr(st(cols)), self.ptr(&out));
        launch!(self, "col2im", threads(n); &cp, &op, &p);
        wrap(out)
    }

    // ---- fused ops (last axis); other axes use the composed default ----

    fn softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend(self.ordinal)).softmax(x, shape, axis);
        }
        self.softmax_rows(x, shape, false)
    }

    fn log_softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend(self.ordinal)).log_softmax(x, shape, axis);
        }
        self.softmax_rows(x, shape, true)
    }

    fn softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend(self.ordinal)).softmax_backward(y, g, shape, axis);
        }
        self.softmax_backward_rows(y, g, shape, false)
    }

    fn log_softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend(self.ordinal)).log_softmax_backward(y, g, shape, axis);
        }
        self.softmax_backward_rows(y, g, shape, true)
    }

    fn layer_norm(&self, x: &Storage, r: usize, dim: usize, eps: f32) -> (Storage, Storage) {
        let xhat = self.alloc(r * dim, DType::F32);
        let rstd = self.alloc(r, DType::F32);
        let (xp, hp, rp, d) = (self.ptr(st(x)), self.ptr(&xhat), self.ptr(&rstd), u(dim));
        launch!(self, "layer_norm_rows", rows(r, dim); &xp, &hp, &rp, &d, &eps);
        (wrap(xhat), wrap(rstd))
    }

    fn layer_norm_backward(&self, g: &Storage, xhat: &Storage, rstd: &Storage, r: usize, dim: usize) -> Storage {
        let out = self.alloc(r * dim, DType::F32);
        let (gp, hp, rp, op, d) = (self.ptr(st(g)), self.ptr(st(xhat)), self.ptr(st(rstd)), self.ptr(&out), u(dim));
        launch!(self, "layer_norm_backward_rows", rows(r, dim); &gp, &hp, &rp, &op, &d);
        wrap(out)
    }

    fn adam_step(&self, p: &Storage, g: &Storage, m: &Storage, v: &Storage, h: AdamParams) -> (Storage, Storage, Storage) {
        let n = st(p).len;
        let (np, nm, nv) = (self.alloc(n, DType::F32), self.alloc(n, DType::F32), self.alloc(n, DType::F32));
        let k = AdamKernelParams {
            n: u(n),
            lr: h.lr,
            beta1: h.beta1,
            beta2: h.beta2,
            eps: h.eps,
            wd: h.weight_decay,
            decoupled: h.decoupled as u32,
            ib1: 1.0 / h.bias_correction1,
            ib2: 1.0 / h.bias_correction2,
        };
        let ptrs = [st(p), st(g), st(m), st(v), &np, &nm, &nv].map(|s| self.ptr(s));
        let [pp, gp, mp, vp, npp, nmp, nvp] = ptrs;
        launch!(self, "adam_step", threads(n); &pp, &gp, &mp, &vp, &npp, &nmp, &nvp, &k);
        (wrap(np), wrap(nm), wrap(nv))
    }
}
