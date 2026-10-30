//! Apple Metal backend (`--features mps`, macOS 15+ / Apple Silicon).
//!
//! * Storage: `MTLBuffer`s in shared (unified) memory, so host transfers are memcpys.
//! * Kernels: `mps_kernels.metal`, compiled at runtime with safe math.
//! * Matmul: `MPSMatrixMultiplication` (batched, with transposes).
//! * Ops are encoded into a shared command buffer and only submitted every few
//!   dozen dispatches or when the host needs data (`to_f32`/`to_i32`), so the
//!   CPU never waits per op. Metal's automatic hazard tracking orders the work.
//! * Kernel outputs come from a size-keyed buffer pool to avoid allocations.

use super::{AdamParams, Backend, BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_strides, numel, split_axis, strides};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CmdBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
/// (trans_a, trans_b, m, n, k) -> cached GEMM kernel
type GemmCache = HashMap<(bool, bool, usize, usize, usize), Retained<MPSMatrixMultiplication>>;

const KERNELS: &str = include_str!("mps_kernels.metal");
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
/// Submit the command buffer after this many encoded ops.
const OPS_PER_COMMIT: usize = 48;
/// Upper bound on idle bytes kept in the buffer pool.
const POOL_LIMIT_BYTES: usize = 1 << 30;

/// A pooled Metal buffer; returns itself to the pool when the last storage drops.
pub struct BufferHandle {
    buf: Option<MtlBuffer>,
    bytes: usize,
}

// Metal buffers are thread safe; all encoding is serialized by `MpsBackend::stream`.
unsafe impl Send for BufferHandle {}
unsafe impl Sync for BufferHandle {}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        if let (Some(buf), Some(be)) = (self.buf.take(), BACKEND.get()) {
            be.pool.lock().unwrap().put(self.bytes, buf);
        }
    }
}

impl BufferHandle {
    fn mtl(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.buf.as_ref().unwrap()
    }
}

/// Device buffer: shared-memory `MTLBuffer` + element count and dtype.
#[derive(Clone)]
pub struct MpsStorage {
    buf: Arc<BufferHandle>,
    pub len: usize,
    pub dtype: DType,
}

#[derive(Default)]
struct Pool {
    free: HashMap<usize, Vec<MtlBuffer>>,
    bytes: usize,
}

impl Pool {
    fn take(&mut self, bytes: usize) -> Option<MtlBuffer> {
        let b = self.free.get_mut(&bytes)?.pop()?;
        self.bytes -= bytes;
        Some(b)
    }

    fn put(&mut self, bytes: usize, buf: MtlBuffer) {
        if self.bytes + bytes <= POOL_LIMIT_BYTES {
            self.bytes += bytes;
            self.free.entry(bytes).or_default().push(buf);
        }
    }
}

#[derive(Default)]
struct Stream {
    cmd: Option<CmdBuffer>,
    enc: Option<Encoder>,
    ops: usize,
    last: Option<CmdBuffer>,
}

impl Stream {
    fn cmd(&mut self, queue: &ProtocolObject<dyn MTLCommandQueue>) -> CmdBuffer {
        if self.cmd.is_none() {
            self.cmd = Some(queue.commandBuffer().expect("failed to create Metal command buffer"));
        }
        self.cmd.clone().unwrap()
    }

    fn encoder(&mut self, queue: &ProtocolObject<dyn MTLCommandQueue>) -> Encoder {
        if self.enc.is_none() {
            let cmd = self.cmd(queue);
            self.enc = Some(cmd.computeCommandEncoder().expect("failed to create compute encoder"));
        }
        self.enc.clone().unwrap()
    }

    fn end_encoder(&mut self) {
        if let Some(enc) = self.enc.take() {
            enc.endEncoding();
        }
    }

    /// Submit pending work; with `wait`, block until all submitted work is done.
    fn commit(&mut self, wait: bool) {
        self.end_encoder();
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            self.last = Some(cmd);
        }
        self.ops = 0;
        if wait {
            if let Some(cmd) = self.last.take() {
                cmd.waitUntilCompleted();
                if cmd.status() == MTLCommandBufferStatus::Error {
                    panic!("Metal command buffer failed: {:?}", cmd.error());
                }
            }
        }
    }
}

pub struct MpsBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Pipeline>,
    stream: Mutex<Stream>,
    pool: Mutex<Pool>,
    gemms: Mutex<GemmCache>,
}

unsafe impl Send for MpsBackend {}
unsafe impl Sync for MpsBackend {}

static BACKEND: OnceLock<MpsBackend> = OnceLock::new();

/// Whether a Metal GPU is present.
pub fn is_available() -> bool {
    BACKEND.get().is_some() || MTLCreateSystemDefaultDevice().is_some()
}

pub fn backend() -> &'static dyn Backend {
    BACKEND.get_or_init(MpsBackend::new)
}

enum Arg<'a> {
    Buf(&'a MpsStorage),
    Bytes(&'a [u8]),
}

enum Grid {
    /// One thread per element.
    Threads(usize),
    /// `rows` threadgroups of `tg` threads.
    Rows { rows: usize, tg: usize },
}

fn bytes_of<T>(v: &T) -> &[u8] {
    // Safety: only used with #[repr(C)] plain-old-data parameter structs / scalars.
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("MPS backend: size exceeds u32")
}

fn st(s: &Storage) -> &MpsStorage {
    match s {
        Storage::Mps(m) => m,
        #[allow(unreachable_patterns)]
        _ => panic!("MPS backend received a non-MPS storage ({})", s.device()),
    }
}

fn wrap(s: MpsStorage) -> Storage {
    Storage::Mps(s)
}

/// Threads per threadgroup for row kernels: a multiple of 32, at most 256.
fn row_threads(d: usize) -> usize {
    d.div_ceil(32).clamp(1, 8) * 32
}

#[repr(C)]
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
struct CopyParams {
    n: u32,
    rank: u32,
    base: u32,
    pad: u32,
    shape: [u32; MAX_RANK],
    strides: [u32; MAX_RANK],
}

#[repr(C)]
struct CatParams {
    n: u32,
    blk_in: u32,
    blk_out: u32,
    offset: u32,
}

#[repr(C)]
struct RedParams {
    outer: u32,
    len: u32,
    inner: u32,
    op: u32,
}

#[repr(C)]
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
struct GatherParams {
    n: u32,
    len: u32,
    idx_len: u32,
    inner: u32,
}

#[repr(C)]
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

fn fill8(v: &[usize]) -> [u32; MAX_RANK] {
    assert!(v.len() <= MAX_RANK, "MPS backend supports tensors of rank <= {MAX_RANK}");
    let mut a = [0u32; MAX_RANK];
    for (d, &x) in a.iter_mut().zip(v) {
        *d = u(x);
    }
    a
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

impl MpsBackend {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device available");
        let queue = device.newCommandQueue().expect("failed to create Metal command queue");
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNELS), Some(&options))
            .unwrap_or_else(|e| panic!("failed to compile Metal kernels: {e:?}"));
        let pipelines = KERNEL_NAMES
            .iter()
            .map(|&name| {
                let f = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .unwrap_or_else(|| panic!("Metal kernel `{name}` not found"));
                let p = device
                    .newComputePipelineStateWithFunction_error(&f)
                    .unwrap_or_else(|e| panic!("failed to build pipeline `{name}`: {e:?}"));
                (name, p)
            })
            .collect();
        MpsBackend {
            device,
            queue,
            pipelines,
            stream: Mutex::new(Stream::default()),
            pool: Mutex::new(Pool::default()),
            gemms: Mutex::new(HashMap::new()),
        }
    }

    /// Uninitialized storage for `len` elements (from the pool when possible).
    fn alloc(&self, len: usize, dtype: DType) -> MpsStorage {
        let bytes = len.max(1) * 4;
        let buf = self.pool.lock().unwrap().take(bytes).unwrap_or_else(|| {
            self.device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .expect("Metal buffer allocation failed")
        });
        MpsStorage { buf: Arc::new(BufferHandle { buf: Some(buf), bytes }), len, dtype }
    }

    fn upload(&self, ptr: *const c_void, len: usize, dtype: DType) -> MpsStorage {
        let bytes = len.max(1) * 4;
        // a fresh buffer: pooled ones may still be in use by queued GPU work
        let buf = if len == 0 {
            self.device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        } else {
            unsafe {
                self.device.newBufferWithBytes_length_options(
                    NonNull::new(ptr as *mut c_void).unwrap(),
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                )
            }
        }
        .expect("Metal buffer allocation failed");
        MpsStorage { buf: Arc::new(BufferHandle { buf: Some(buf), bytes }), len, dtype }
    }

    /// Wait for queued work, then view the buffer's 32-bit words on the host.
    fn read<T: Copy>(&self, s: &MpsStorage) -> Vec<T> {
        self.stream.lock().unwrap().commit(true);
        let p = s.buf.mtl().contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(p, s.len).to_vec() }
    }

    fn run(&self, kernel: &str, args: &[Arg], grid: Grid) {
        let n = match grid {
            Grid::Threads(n) => n,
            Grid::Rows { rows, .. } => rows,
        };
        if n == 0 {
            return;
        }
        let pso = &self.pipelines[kernel];
        let mut stream = self.stream.lock().unwrap();
        let enc = stream.encoder(&self.queue);
        enc.setComputePipelineState(pso);
        for (i, a) in args.iter().enumerate() {
            unsafe {
                match a {
                    Arg::Buf(s) => enc.setBuffer_offset_atIndex(Some(s.buf.mtl()), 0, i),
                    Arg::Bytes(b) => {
                        enc.setBytes_length_atIndex(NonNull::new(b.as_ptr() as *mut c_void).unwrap(), b.len(), i)
                    }
                }
            }
        }
        let one = |w| MTLSize { width: w, height: 1, depth: 1 };
        match grid {
            Grid::Threads(n) => {
                let w = pso.maxTotalThreadsPerThreadgroup().min(256);
                enc.dispatchThreads_threadsPerThreadgroup(one(n), one(w));
            }
            Grid::Rows { rows, tg } => enc.dispatchThreadgroups_threadsPerThreadgroup(one(rows), one(tg)),
        }
        stream.ops += 1;
        if stream.ops >= OPS_PER_COMMIT {
            stream.commit(false);
        }
    }

    fn to_f32_storage(&self, s: &MpsStorage) -> MpsStorage {
        if s.dtype == DType::F32 {
            return s.clone();
        }
        let out = self.alloc(s.len, DType::F32);
        let n = u(s.len);
        self.run("cast_i32_f32", &[Arg::Buf(s), Arg::Buf(&out), Arg::Bytes(bytes_of(&n))], Grid::Threads(s.len));
        out
    }

    fn zeros(&self, len: usize) -> MpsStorage {
        let out = self.alloc(len, DType::F32);
        let (n, bits) = (u(len), 0u32);
        self.run(
            "fill_u32",
            &[Arg::Buf(&out), Arg::Bytes(bytes_of(&n)), Arg::Bytes(bytes_of(&bits))],
            Grid::Threads(len),
        );
        out
    }

    /// Generic strided gather copy (any 32-bit dtype).
    fn strided_copy(&self, x: &MpsStorage, out_shape: &[usize], in_strides: &[usize], base: usize) -> MpsStorage {
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
        self.run("strided_copy", &[Arg::Buf(x), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))], Grid::Threads(n));
        out
    }

    fn gemm_kernel(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize) -> Retained<MPSMatrixMultiplication> {
        let mut cache = self.gemms.lock().unwrap();
        cache
            .entry((ta, tb, m, n, k))
            .or_insert_with(|| unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.device,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    1.0,
                    0.0,
                )
            })
            .clone()
    }
}

fn matrix(s: &MpsStorage, rows: usize, cols: usize, batch: usize) -> Retained<MPSMatrix> {
    unsafe {
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
            rows,
            cols,
            batch,
            cols * 4,
            rows * cols * 4,
            MPSDataType::Float32,
        );
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), s.buf.mtl(), &desc)
    }
}

impl Backend for MpsBackend {
    fn device(&self) -> Device {
        Device::Mps
    }

    fn synchronize(&self) {
        self.stream.lock().unwrap().commit(true);
    }

    fn from_f32(&self, data: Vec<f32>) -> Storage {
        wrap(self.upload(data.as_ptr() as *const c_void, data.len(), DType::F32))
    }

    fn from_i32(&self, data: Vec<i32>) -> Storage {
        wrap(self.upload(data.as_ptr() as *const c_void, data.len(), DType::I32))
    }

    fn to_f32(&self, x: &Storage) -> Vec<f32> {
        let s = st(x);
        match s.dtype {
            DType::F32 => self.read::<f32>(s),
            DType::I32 => self.read::<i32>(s).into_iter().map(|v| v as f32).collect(),
        }
    }

    fn to_i32(&self, x: &Storage) -> Vec<i32> {
        let s = st(x);
        match s.dtype {
            DType::I32 => self.read::<i32>(s),
            DType::F32 => self.read::<f32>(s).into_iter().map(|v| v as i32).collect(),
        }
    }

    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage {
        let out = self.alloc(len, dtype);
        let bits = match dtype {
            DType::F32 => value.to_bits(),
            DType::I32 => (value as i32) as u32,
        };
        let n = u(len);
        self.run(
            "fill_u32",
            &[Arg::Buf(&out), Arg::Bytes(bytes_of(&n)), Arg::Bytes(bytes_of(&bits))],
            Grid::Threads(len),
        );
        wrap(out)
    }

    fn cast(&self, x: &Storage, dtype: DType) -> Storage {
        let s = st(x);
        if s.dtype == dtype {
            return x.clone();
        }
        let out = self.alloc(s.len, dtype);
        let kernel = if dtype == DType::I32 { "cast_f32_i32" } else { "cast_i32_f32" };
        let n = u(s.len);
        self.run(kernel, &[Arg::Buf(s), Arg::Buf(&out), Arg::Bytes(bytes_of(&n))], Grid::Threads(s.len));
        wrap(out)
    }

    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
        let s = st(x);
        assert_eq!(s.dtype, DType::F32, "MPS unary op on {}", s.dtype);
        let out = self.alloc(s.len, DType::F32);
        let (n, code) = (u(s.len), unary_code(op));
        self.run(
            "unary_f32",
            &[Arg::Buf(s), Arg::Buf(&out), Arg::Bytes(bytes_of(&n)), Arg::Bytes(bytes_of(&code))],
            Grid::Threads(s.len),
        );
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
        self.run(
            "binary_f32",
            &[Arg::Buf(&a), Arg::Buf(&b), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))],
            Grid::Threads(n),
        );
        wrap(out)
    }

    fn expand(&self, x: &Storage, shape: &[usize], out_shape: &[usize]) -> Storage {
        wrap(self.strided_copy(st(x), out_shape, &broadcast_strides(shape, out_shape), 0))
    }

    fn where_cond(&self, cond: &Storage, a: &Storage, b: &Storage) -> Storage {
        let c = self.to_f32_storage(st(cond));
        let (a, b) = (st(a), st(b));
        let out = self.alloc(a.len, DType::F32);
        let n = u(a.len);
        self.run(
            "where_f32",
            &[Arg::Buf(&c), Arg::Buf(a), Arg::Buf(b), Arg::Buf(&out), Arg::Bytes(bytes_of(&n))],
            Grid::Threads(a.len),
        );
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
        let args = [Arg::Buf(s), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))];
        if inner == 1 && op != ReduceOp::ArgMax {
            self.run("reduce_rows", &args, Grid::Rows { rows: outer, tg: row_threads(len) });
        } else {
            self.run("reduce_strided", &args, Grid::Threads(outer * inner));
        }
        wrap(out)
    }

    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
        let MatmulSpec { batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride } = spec;
        if k == 0 {
            return wrap(self.zeros(batch * m * n));
        }
        // MPS needs a real batch stride: materialize broadcast operands.
        let bcast = |s: &MpsStorage, stride: usize, sz: usize| {
            if batch > 1 && stride == 0 {
                self.strided_copy(s, &[batch, sz], &[0, 1], 0)
            } else {
                s.clone()
            }
        };
        let a = bcast(st(a), a_batch_stride, m * k);
        let b = bcast(st(b), b_batch_stride, k * n);
        let out = self.alloc(batch * m * n, DType::F32);
        let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
        let (br, bc) = if trans_b { (n, k) } else { (k, n) };
        let (ma, mb, mc) = (matrix(&a, ar, ac, batch), matrix(&b, br, bc, batch), matrix(&out, m, n, batch));
        let kernel = self.gemm_kernel(trans_a, trans_b, m, n, k);
        let mut stream = self.stream.lock().unwrap();
        stream.end_encoder();
        let cmd = stream.cmd(&self.queue);
        unsafe {
            kernel.setBatchStart(0);
            kernel.setBatchSize(batch);
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&cmd, &ma, &mb, &mc);
        }
        stream.ops += 1;
        if stream.ops >= OPS_PER_COMMIT {
            stream.commit(false);
        }
        wrap(out)
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
        let mut offset = 0;
        for (x, s) in xs.iter().zip(shapes) {
            let blk_in = s[axis] * inner;
            let x = st(x);
            let p = CatParams { n: u(x.len), blk_in: u(blk_in), blk_out: u(blk_out), offset: u(offset) };
            self.run("cat_copy", &[Arg::Buf(x), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))], Grid::Threads(x.len));
            offset += blk_in;
        }
        wrap(out)
    }

    fn index_select(&self, table: &Storage, cols: usize, idx: &Storage) -> Storage {
        let (t, idx) = (st(table), st(idx));
        let n = idx.len * cols;
        let out = self.alloc(n, DType::F32);
        let (nn, c) = (u(n), u(cols));
        self.run(
            "index_select",
            &[Arg::Buf(t), Arg::Buf(idx), Arg::Buf(&out), Arg::Bytes(bytes_of(&nn)), Arg::Bytes(bytes_of(&c))],
            Grid::Threads(n),
        );
        wrap(out)
    }

    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &Storage) -> Storage {
        let (s, idx) = (st(src), st(idx));
        let out = self.zeros(rows * cols);
        let (nn, c) = (u(s.len), u(cols));
        self.run(
            "index_add",
            &[Arg::Buf(s), Arg::Buf(idx), Arg::Buf(&out), Arg::Bytes(bytes_of(&nn)), Arg::Bytes(bytes_of(&c))],
            Grid::Threads(s.len),
        );
        wrap(out)
    }

    fn gather(&self, x: &Storage, shape: &[usize], axis: usize, idx: &Storage, idx_len: usize) -> Storage {
        let (x, idx) = (st(x), st(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.alloc(idx.len, DType::F32);
        let p = GatherParams { n: u(idx.len), len: u(len), idx_len: u(idx_len), inner: u(inner) };
        self.run(
            "gather",
            &[Arg::Buf(x), Arg::Buf(idx), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))],
            Grid::Threads(idx.len),
        );
        wrap(out)
    }

    fn scatter_add(&self, src: &Storage, idx: &Storage, idx_len: usize, shape: &[usize], axis: usize) -> Storage {
        let (s, idx) = (st(src), st(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.zeros(numel(shape));
        let p = GatherParams { n: u(idx.len), len: u(len), idx_len: u(idx_len), inner: u(inner) };
        self.run(
            "scatter_add",
            &[Arg::Buf(s), Arg::Buf(idx), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))],
            Grid::Threads(idx.len),
        );
        wrap(out)
    }

    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
        let (oh, ow) = g.out_hw();
        let n = g.n * g.c * g.kh * g.kw * oh * ow;
        let out = self.alloc(n, DType::F32);
        let p = conv_params(g);
        self.run("im2col", &[Arg::Buf(st(x)), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))], Grid::Threads(n));
        wrap(out)
    }

    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage {
        let n = g.n * g.c * g.h * g.w;
        let out = self.alloc(n, DType::F32);
        let p = conv_params(g);
        self.run("col2im", &[Arg::Buf(st(cols)), Arg::Buf(&out), Arg::Bytes(bytes_of(&p))], Grid::Threads(n));
        wrap(out)
    }

    // ---- fused ops (last axis); other axes use the composed default ----

    fn softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).softmax(x, shape, axis);
        }
        self.softmax_rows(x, shape, false)
    }

    fn log_softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).log_softmax(x, shape, axis);
        }
        self.softmax_rows(x, shape, true)
    }

    fn softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).softmax_backward(y, g, shape, axis);
        }
        self.softmax_backward_rows(y, g, shape, false)
    }

    fn log_softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).log_softmax_backward(y, g, shape, axis);
        }
        self.softmax_backward_rows(y, g, shape, true)
    }

    fn layer_norm(&self, x: &Storage, rows: usize, dim: usize, eps: f32) -> (Storage, Storage) {
        let xhat = self.alloc(rows * dim, DType::F32);
        let rstd = self.alloc(rows, DType::F32);
        let d = u(dim);
        self.run(
            "layer_norm_rows",
            &[Arg::Buf(st(x)), Arg::Buf(&xhat), Arg::Buf(&rstd), Arg::Bytes(bytes_of(&d)), Arg::Bytes(bytes_of(&eps))],
            Grid::Rows { rows, tg: row_threads(dim) },
        );
        (wrap(xhat), wrap(rstd))
    }

    fn layer_norm_backward(&self, g: &Storage, xhat: &Storage, rstd: &Storage, rows: usize, dim: usize) -> Storage {
        let out = self.alloc(rows * dim, DType::F32);
        let d = u(dim);
        self.run(
            "layer_norm_backward_rows",
            &[Arg::Buf(st(g)), Arg::Buf(st(xhat)), Arg::Buf(st(rstd)), Arg::Buf(&out), Arg::Bytes(bytes_of(&d))],
            Grid::Rows { rows, tg: row_threads(dim) },
        );
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
        self.run(
            "adam_step",
            &[
                Arg::Buf(st(p)),
                Arg::Buf(st(g)),
                Arg::Buf(st(m)),
                Arg::Buf(st(v)),
                Arg::Buf(&np),
                Arg::Buf(&nm),
                Arg::Buf(&nv),
                Arg::Bytes(bytes_of(&k)),
            ],
            Grid::Threads(n),
        );
        (wrap(np), wrap(nm), wrap(nv))
    }
}

impl MpsBackend {
    fn softmax_rows(&self, x: &Storage, shape: &[usize], log: bool) -> Storage {
        let d = shape[shape.len() - 1];
        let rows = numel(shape) / d.max(1);
        let out = self.alloc(rows * d, DType::F32);
        let (du, lm) = (u(d), log as u32);
        self.run(
            "softmax_rows",
            &[Arg::Buf(st(x)), Arg::Buf(&out), Arg::Bytes(bytes_of(&du)), Arg::Bytes(bytes_of(&lm))],
            Grid::Rows { rows, tg: row_threads(d) },
        );
        wrap(out)
    }

    fn softmax_backward_rows(&self, y: &Storage, g: &Storage, shape: &[usize], log: bool) -> Storage {
        let d = shape[shape.len() - 1];
        let rows = numel(shape) / d.max(1);
        let out = self.alloc(rows * d, DType::F32);
        let (du, lm) = (u(d), log as u32);
        self.run(
            "softmax_backward_rows",
            &[Arg::Buf(st(y)), Arg::Buf(st(g)), Arg::Buf(&out), Arg::Bytes(bytes_of(&du)), Arg::Bytes(bytes_of(&lm))],
            Grid::Rows { rows, tg: row_threads(d) },
        );
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
