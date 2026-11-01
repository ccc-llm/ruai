//! WebGPU backend (`--features webgpu`), on wgpu: Metal, Vulkan or DX12
//! natively, the browser's WebGPU on wasm.
//!
//! * Kernels: WGSL (`webgpu_kernels.wgsl`), one module per kernel, compiled on
//!   first use; parameters are u32 words in a ring buffer (read-only storage).
//! * Storage is 32-bit words: f32, i32, half floats as f32 values rounded to
//!   f16 / bf16 (every op computes in f32 and rounds its result, like the
//!   CPU backend), i8 as packed bytes.
//! * Matmul: a tiled WGSL GEMM generated with its register tile unrolled
//!   (128 x 128 tiles, 8 x 8 per thread; transposes and batch strides read in place).
//! * Ops are encoded into one command encoder, submitted every few dozen
//!   dispatches or when the host reads data; outputs come from a size-keyed
//!   buffer pool.
//! * Reading data back waits for the GPU. Natively that blocks (`Device::poll`).
//!   A browser cannot block: there the backend is created by [`init`]`.await`
//!   and data read with [`crate::nn::Tensor::to_vec_async`]`.await`.

use super::{AdamParams, Backend, BinaryOp, Conv2dGeom, DType, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::nn::device::Device;
use crate::nn::shape::{broadcast_strides, numel, split_axis, strides};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;

const KERNELS: &str = include_str!("webgpu_kernels.wgsl");
const MAX_RANK: usize = 8;
/// Submit the command encoder after this many dispatches.
const OPS_PER_SUBMIT: usize = 64;
/// Bytes of the parameter ring buffer.
const PARAMS_BYTES: u64 = 1 << 20;
/// Upper bound on idle bytes kept in the buffer pool.
const POOL_LIMIT_BYTES: usize = 1 << 30;
/// Threads per workgroup of the elementwise and row kernels.
const WG: usize = 256;

/// A pooled buffer; returns itself to the pool when the last storage drops.
pub struct BufferHandle {
    buf: Option<wgpu::Buffer>,
    bytes: usize,
}

// wasm32-unknown-unknown has no threads: wgpu's browser objects (not Send /
// Sync) are only ever used by the one thread
#[cfg(target_arch = "wasm32")]
unsafe impl Send for BufferHandle {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for BufferHandle {}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        if let (Some(buf), Some(Some(be))) = (self.buf.take(), BACKEND.get()) {
            be.pool.lock().unwrap().put(self.bytes, buf);
        }
    }
}

/// Device buffer + element count and dtype.
#[derive(Clone)]
pub struct WebGpuStorage {
    buf: Arc<BufferHandle>,
    /// Number of elements.
    pub len: usize,
    /// Element type.
    pub dtype: DType,
}

impl WebGpuStorage {
    fn gpu(&self) -> &wgpu::Buffer {
        self.buf.buf.as_ref().unwrap()
    }
    /// The same buffer with another dtype label (f32 and half floats share
    /// their representation).
    fn relabel(&self, dtype: DType) -> WebGpuStorage {
        WebGpuStorage { buf: self.buf.clone(), len: self.len, dtype }
    }
}

#[derive(Default)]
struct Pool {
    free: HashMap<usize, Vec<wgpu::Buffer>>,
    bytes: usize,
}

impl Pool {
    fn take(&mut self, bytes: usize) -> Option<wgpu::Buffer> {
        let b = self.free.get_mut(&bytes)?.pop()?;
        self.bytes -= bytes;
        Some(b)
    }
    fn put(&mut self, bytes: usize, buf: wgpu::Buffer) {
        if self.bytes + bytes <= POOL_LIMIT_BYTES {
            self.bytes += bytes;
            self.free.entry(bytes).or_default().push(buf);
        }
    }
}

#[derive(Default)]
struct Stream {
    encoder: Option<wgpu::CommandEncoder>,
    /// the open compute pass of `encoder` (dispatches share one pass)
    pass: Option<wgpu::ComputePass<'static>>,
    ops: usize,
    /// next free offset in the parameter ring
    params: u64,
}

/// [`Backend`] implementation on WebGPU.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    pipelines: Mutex<HashMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
    stream: Mutex<Stream>,
    pool: Mutex<Pool>,
    params: wgpu::Buffer,
    align: u64,
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WebGpuBackend {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WebGpuBackend {}

static BACKEND: OnceLock<Option<WebGpuBackend>> = OnceLock::new();

fn get() -> Option<&'static WebGpuBackend> {
    // natively created on first use; in a browser only by `init().await`
    #[cfg(not(target_arch = "wasm32"))]
    return BACKEND.get_or_init(|| pollster::block_on(WebGpuBackend::new())).as_ref();
    #[cfg(target_arch = "wasm32")]
    return BACKEND.get().and_then(|b| b.as_ref());
}

/// Create the backend without blocking (needed in a browser, where adapter
/// and device requests are asynchronous); `true` if a WebGPU adapter exists.
pub async fn init() -> bool {
    if BACKEND.get().is_none() {
        let b = WebGpuBackend::new().await;
        let _ = BACKEND.set(b);
    }
    BACKEND.get().is_some_and(|b| b.is_some())
}

/// A value sent by a callback, awaited by a future (wgpu's buffer mapping).
struct Slot<T>(Arc<Mutex<(Option<T>, Option<std::task::Waker>)>>);

impl<T> std::future::Future for Slot<T> {
    type Output = T;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<T> {
        let mut g = self.0.lock().unwrap();
        match g.0.take() {
            Some(v) => std::task::Poll::Ready(v),
            None => {
                g.1 = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
    }
}

/// A storage's f32 values, read without blocking.
pub(crate) async fn to_f32_async(x: &Storage) -> Vec<f32> {
    let be = get().expect("no WebGPU adapter available");
    let s = st(x);
    let b = be.read_bytes_async(s).await;
    match s.dtype {
        DType::F32 | DType::F16 | DType::BF16 => b.chunks_exact(4).take(s.len).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        DType::I32 => b.chunks_exact(4).take(s.len).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
        DType::I8 => b[..s.len].iter().map(|&v| v as i8 as f32).collect(),
    }
}

/// Whether a WebGPU adapter is present.
pub fn is_available() -> bool {
    get().is_some()
}

/// The process-wide WebGPU backend (created on first use).
pub fn backend() -> &'static dyn Backend {
    get().expect("no WebGPU adapter available")
}

/// The name of the adapter in use (for example "Apple M3 (Metal)").
pub fn adapter_name() -> Option<String> {
    get().map(|b| b.adapter_name.clone())
}

fn st(s: &Storage) -> &WebGpuStorage {
    match s {
        Storage::WebGpu(m) => m,
        #[allow(unreachable_patterns)]
        _ => panic!("WebGPU backend received a non-WebGPU storage ({})", s.device()),
    }
}

fn wrap(s: WebGpuStorage) -> Storage {
    Storage::WebGpu(s)
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("WebGPU backend: size exceeds u32")
}

fn fill8(v: &[usize]) -> [u32; MAX_RANK] {
    assert!(v.len() <= MAX_RANK, "WebGPU backend supports tensors of rank <= {MAX_RANK}");
    let mut a = [0u32; MAX_RANK];
    for (d, &x) in a.iter_mut().zip(v) {
        *d = u(x);
    }
    a
}

/// Words of a buffer of `len` elements of `dtype` (i8: four per word).
fn words(len: usize, dtype: DType) -> usize {
    if dtype == DType::I8 {
        len.div_ceil(4)
    } else {
        len
    }
}

/// Workgroups for `n` threads of 256, on a 2-D grid when there are more than 65535.
fn grid(n: usize) -> (u32, u32, u32) {
    let groups = n.div_ceil(WG).max(1);
    if groups <= 65535 {
        (groups as u32, 1, 1)
    } else {
        (65535, groups.div_ceil(65535) as u32, 1)
    }
}

/// One workgroup per row.
fn rows_grid(rows: usize) -> (u32, u32, u32) {
    let rows = rows.max(1);
    if rows <= 65535 {
        (rows as u32, 1, 1)
    } else {
        (65535, rows.div_ceil(65535) as u32, 1)
    }
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

/// The GEMM kernel, generated with its register tile unrolled (WGSL has no
/// unrolling, and arrays indexed in loops would live in memory): a 128 x 128
/// tile of C per workgroup of 16 x 16 threads, each 8 x 8 in 16 vec4
/// registers, k in steps of 8 through workgroup memory.
/// p: batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride
fn gemm_source(prelude: &str) -> String {
    use std::fmt::Write;
    let mut s = String::from(prelude);
    s += "@group(0) @binding(1) var<storage, read> A: array<f32>;\n\
          @group(0) @binding(2) var<storage, read> B: array<f32>;\n\
          @group(0) @binding(3) var<storage, read_write> C: array<f32>;\n\
          var<workgroup> As: array<f32, 1024>;\n\
          var<workgroup> Bs: array<vec4<f32>, 256>;\n\
          @compute @workgroup_size(16, 16)\n\
          fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n\
          let M = p[1]; let K = p[2]; let N = p[3];\n\
          let ta = p[4] != 0u; let tb = p[5] != 0u;\n\
          let aoff = wg.z * p[6]; let boff = wg.z * p[7];\n\
          let row0 = wg.y * 128u; let col0 = wg.x * 128u;\n\
          let t = lid.y * 16u + lid.x;\n";
    for i in 0..8 {
        let _ = writeln!(s, "var c{i}a = vec4<f32>(0.0); var c{i}b = vec4<f32>(0.0);");
    }
    s += "for (var k0 = 0u; k0 < K; k0 += 8u) {\n";
    // A tile: 128 rows x 8 k, 4 per thread (coalesced along k, or along m when transposed)
    for e in 0..4 {
        let _ = writeln!(
            s,
            "{{ let idx = t + {o}u; var r = idx / 8u; var ka = idx % 8u; if (ta) {{ r = idx % 128u; ka = idx / 128u; }}\n\
             let gr = row0 + r; let gk = k0 + ka; var av = 0.0;\n\
             if (gr < M && gk < K) {{ av = A[aoff + select(gr * K + gk, gk * M + gr, ta)]; }}\n\
             As[ka * 128u + r] = av; }}",
            o = e * 256
        );
    }
    // B tile: 8 k x 128 columns, one vec4 per thread
    s += "{ let kb = t / 32u; let c4 = t % 32u; let gk = k0 + kb; var bv = vec4<f32>(0.0);\n\
          if (gk < K) {\n";
    for j in 0..4 {
        let _ = writeln!(s, "{{ let gc = col0 + c4 * 4u + {j}u; if (gc < N) {{ bv[{j}] = B[boff + select(gk * N + gc, gc * K + gk, tb)]; }} }}");
    }
    s += "}\nBs[kb * 32u + c4] = bv; }\nworkgroupBarrier();\n\
          for (var kk = 0u; kk < 8u; kk++) {\n\
          let b0 = Bs[kk * 32u + lid.x]; let b1 = Bs[kk * 32u + 16u + lid.x];\n\
          let ab = kk * 128u + lid.y * 8u;\n";
    for i in 0..8 {
        let _ = writeln!(s, "let a{i} = As[ab + {i}u]; c{i}a += a{i} * b0; c{i}b += a{i} * b1;");
    }
    s += "}\nworkgroupBarrier();\n}\n\
          let coff = wg.z * M * N; let r = row0 + lid.y * 8u;\n\
          let ca = col0 + lid.x * 4u; let cb = col0 + 64u + lid.x * 4u;\n";
    for i in 0..8 {
        let _ = writeln!(s, "if (r + {i}u < M) {{ let o = coff + (r + {i}u) * N; for (var j = 0u; j < 4u; j++) {{ if (ca + j < N) {{ C[o + ca + j] = c{i}a[j]; }} if (cb + j < N) {{ C[o + cb + j] = c{i}b[j]; }} }} }}");
    }
    s += "}\n";
    s
}

/// The source of kernel `name`: the prelude and its section.
fn kernel_source(name: &str) -> String {
    // sections start at a line `//! kernel: NAME`
    let marker = "\n//! kernel: ";
    let first = KERNELS.find(marker).expect("kernels");
    let prelude = &KERNELS[..first + 1];
    if name == "matmul" {
        return gemm_source(prelude);
    }
    for sec in KERNELS[first..].split(marker).filter(|s| !s.trim().is_empty()) {
        let (n, body) = sec.split_once('\n').unwrap_or((sec, ""));
        if n.trim() == name {
            return format!("{}{}", prelude, body);
        }
    }
    panic!("no WebGPU kernel `{name}`")
}

impl WebGpuBackend {
    async fn new() -> Option<WebGpuBackend> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .ok()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { label: Some("ruai"), required_limits: limits, ..Default::default() }).await.ok()?;
        let align = device.limits().min_storage_buffer_offset_alignment as u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ruai params"),
            size: PARAMS_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(WebGpuBackend {
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            pipelines: Mutex::new(HashMap::new()),
            stream: Mutex::new(Stream::default()),
            pool: Mutex::new(Pool::default()),
            params,
            align,
        })
    }

    fn pipeline(&self, name: &str) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let mut ps = self.pipelines.lock().unwrap();
        if let Some(p) = ps.get(name) {
            return p.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(kernel_source(name).into()),
        });
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = p.get_bind_group_layout(0);
        ps.insert(name.to_string(), (p.clone(), layout.clone()));
        (p, layout)
    }

    /// Uninitialized storage for `len` elements (from the pool when possible).
    fn alloc(&self, len: usize, dtype: DType) -> WebGpuStorage {
        let bytes = words(len, dtype).max(1) * 4;
        let buf = self.pool.lock().unwrap().take(bytes).unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        WebGpuStorage { buf: Arc::new(BufferHandle { buf: Some(buf), bytes }), len, dtype }
    }

    /// A new buffer with these bytes (padded to whole words).
    fn upload(&self, bytes: &[u8], len: usize, dtype: DType) -> WebGpuStorage {
        let n = words(len, dtype).max(1) * 4;
        let mut data = bytes.to_vec();
        data.resize(n, 0);
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        WebGpuStorage { buf: Arc::new(BufferHandle { buf: Some(buf), bytes: n }), len, dtype }
    }

    /// Submit the encoded work.
    fn flush(&self, stream: &mut Stream) {
        stream.pass = None;
        if let Some(enc) = stream.encoder.take() {
            self.queue.submit([enc.finish()]);
        }
        stream.ops = 0;
        stream.params = 0;
    }

    /// Copy a buffer into a mappable staging buffer and submit.
    fn stage(&self, s: &WebGpuStorage) -> wgpu::Buffer {
        let bytes = s.buf.bytes as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ruai readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut stream = self.stream.lock().unwrap();
        stream.pass = None;
        let enc = stream.encoder.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(s.gpu(), 0, &staging, 0, bytes);
        self.flush(&mut stream);
        staging
    }

    /// Map a staging buffer: a future of the mapping's result.
    fn map(&self, staging: &wgpu::Buffer) -> Slot<Result<(), wgpu::BufferAsyncError>> {
        let slot = Arc::new(Mutex::new((None, None::<std::task::Waker>)));
        let s2 = slot.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let mut g = s2.lock().unwrap();
            g.0 = Some(r);
            if let Some(w) = g.1.take() {
                w.wake();
            }
        });
        Slot(slot)
    }

    fn unmap(staging: wgpu::Buffer) -> Vec<u8> {
        let out = staging.slice(..).get_mapped_range().expect("WebGPU mapped range").to_vec();
        staging.unmap();
        out
    }

    /// Wait for the GPU, then copy a buffer's bytes to the host.
    fn read_bytes(&self, s: &WebGpuStorage) -> Vec<u8> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = s;
            panic!("WebGPU in a browser cannot read data back synchronously: use Tensor::to_vec_async().await");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let staging = self.stage(s);
            let mapped = self.map(&staging);
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("WebGPU device lost");
            pollster::block_on(mapped).expect("WebGPU buffer mapping failed");
            Self::unmap(staging)
        }
    }

    /// Copy a buffer's bytes to the host without blocking.
    async fn read_bytes_async(&self, s: &WebGpuStorage) -> Vec<u8> {
        let staging = self.stage(s);
        let mapped = self.map(&staging);
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("WebGPU device lost");
        mapped.await.expect("WebGPU buffer mapping failed");
        Self::unmap(staging)
    }

    fn read_words<T: Copy + Default>(&self, s: &WebGpuStorage) -> Vec<T> {
        let b = self.read_bytes(s);
        let mut out = vec![T::default(); s.len];
        // Safety: T is a 4-byte plain-old-data type (f32 / i32)
        let n = s.len * std::mem::size_of::<T>();
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out.as_mut_ptr() as *mut u8, n) };
        out
    }

    /// Dispatch `kernel` with parameter words and buffers (bindings 1..).
    fn run(&self, kernel: &str, params: &[u32], bufs: &[&WebGpuStorage], groups: (u32, u32, u32)) {
        if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
            return;
        }
        let (pipe, layout) = self.pipeline(kernel);
        let mut stream = self.stream.lock().unwrap();
        let size = (params.len().max(1) * 4) as u64;
        if stream.params + size > PARAMS_BYTES {
            self.flush(&mut stream);
        }
        let off = stream.params;
        let mut words = params.to_vec();
        if words.is_empty() {
            words.push(0);
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.queue.write_buffer(&self.params, off, &bytes);
        stream.params = (off + size).div_ceil(self.align) * self.align;
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: off, size: wgpu::BufferSize::new(size) }),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.gpu().as_entire_binding() });
        }
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        if stream.pass.is_none() {
            let enc = stream.encoder.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            stream.pass = Some(enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime());
        }
        let pass = stream.pass.as_mut().unwrap();
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        stream.ops += 1;
        if stream.ops >= OPS_PER_SUBMIT {
            self.flush(&mut stream);
        }
    }

    /// An elementwise kernel over `n` items.
    fn each(&self, kernel: &str, params: &[u32], bufs: &[&WebGpuStorage], n: usize) {
        if n > 0 {
            self.run(kernel, params, bufs, grid(n));
        }
    }

    /// f32 values computed for a result of `dtype`: rounded for half floats.
    fn finish(&self, out: WebGpuStorage, dtype: DType) -> WebGpuStorage {
        match dtype {
            DType::F16 | DType::BF16 => {
                let r = self.alloc(out.len, dtype);
                let k = if dtype == DType::F16 { "round_f16" } else { "round_bf16" };
                self.each(k, &[u(out.len)], &[&out, &r], out.len);
                r
            }
            _ => out.relabel(dtype),
        }
    }

    /// A float storage as f32 values (half floats already are; ints converted).
    fn as_f32(&self, s: &WebGpuStorage) -> WebGpuStorage {
        match s.dtype {
            DType::F32 | DType::F16 | DType::BF16 => s.relabel(DType::F32),
            _ => self.convert(s, DType::F32),
        }
    }

    fn convert(&self, s: &WebGpuStorage, dtype: DType) -> WebGpuStorage {
        if s.dtype == dtype {
            return s.clone();
        }
        let n = u(s.len);
        match (s.dtype, dtype) {
            (DType::I8, _) => {
                let to_float = dtype != DType::I32;
                let out = self.alloc(s.len, if to_float { DType::F32 } else { DType::I32 });
                self.each("cast_i8_x", &[n, to_float as u32], &[s, &out], s.len);
                if to_float {
                    self.finish(out, dtype)
                } else {
                    out
                }
            }
            (from, DType::I8) => {
                let out = self.alloc(s.len, DType::I8);
                self.each("cast_x_i8", &[n, (from != DType::I32) as u32], &[s, &out], s.len.div_ceil(4));
                out
            }
            (DType::I32, _) => {
                let out = self.alloc(s.len, DType::F32);
                self.each("cast_i32_f32", &[n], &[s, &out], s.len);
                self.finish(out, dtype)
            }
            (_, DType::I32) => {
                let out = self.alloc(s.len, DType::I32);
                self.each("cast_f32_i32", &[n], &[s, &out], s.len);
                out
            }
            // between float types: f32 values, rounded to the target
            (_, d) => self.finish(s.relabel(DType::F32), d),
        }
    }

    fn zeros(&self, len: usize) -> WebGpuStorage {
        let out = self.alloc(len, DType::F32);
        self.each("fill", &[u(len), 0], &[&out], len);
        out
    }

    /// Generic strided gather copy of 32-bit words (i8 through i32).
    fn strided_copy_raw(&self, x: &WebGpuStorage, out_shape: &[usize], in_strides: &[usize], base: usize) -> WebGpuStorage {
        if x.dtype == DType::I8 {
            let wide = self.convert(x, DType::I32);
            return self.convert(&self.strided_copy_raw(&wide, out_shape, in_strides, base), DType::I8);
        }
        let n = numel(out_shape);
        let out = self.alloc(n, x.dtype);
        let mut p = vec![u(n), u(out_shape.len()), u(base), 0];
        p.extend(fill8(out_shape));
        p.extend(fill8(in_strides));
        self.each("strided_copy", &p, &[x, &out], n);
        out
    }

    fn softmax_rows(&self, x: &WebGpuStorage, shape: &[usize], log: bool) -> WebGpuStorage {
        let d = shape[shape.len() - 1];
        let rows = numel(shape) / d.max(1);
        let out = self.alloc(rows * d, DType::F32);
        self.run("softmax_rows", &[u(rows), u(d), log as u32], &[x, &out], rows_grid(rows));
        out
    }

    fn softmax_backward_rows(&self, y: &WebGpuStorage, g: &WebGpuStorage, shape: &[usize], log: bool) -> WebGpuStorage {
        let d = shape[shape.len() - 1];
        let rows = numel(shape) / d.max(1);
        let out = self.alloc(rows * d, DType::F32);
        self.run("softmax_backward_rows", &[u(rows), u(d), log as u32], &[y, g, &out], rows_grid(rows));
        out
    }
}

fn conv_params(g: Conv2dGeom) -> Vec<u32> {
    let (oh, ow) = g.out_hw();
    [g.n, g.c, g.h, g.w, g.kh, g.kw, g.stride, g.padding, oh, ow].iter().map(|&x| u(x)).collect()
}

impl Backend for WebGpuBackend {
    fn device(&self) -> Device {
        Device::WebGpu
    }

    fn synchronize(&self) {
        let mut stream = self.stream.lock().unwrap();
        self.flush(&mut stream);
        drop(stream);
        // a browser cannot wait: submitting is all it can do
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("WebGPU device lost");
    }

    fn from_f32(&self, data: Vec<f32>) -> Storage {
        let b: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        wrap(self.upload(&b, data.len(), DType::F32))
    }

    fn from_i32(&self, data: Vec<i32>) -> Storage {
        let b: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        wrap(self.upload(&b, data.len(), DType::I32))
    }

    fn from_i8(&self, data: Vec<i8>) -> Storage {
        let b: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        wrap(self.upload(&b, data.len(), DType::I8))
    }

    fn to_f32(&self, x: &Storage) -> Vec<f32> {
        let s = st(x);
        match s.dtype {
            DType::F32 | DType::F16 | DType::BF16 => self.read_words::<f32>(s),
            DType::I32 => self.read_words::<i32>(s).into_iter().map(|v| v as f32).collect(),
            DType::I8 => self.read_bytes(s)[..s.len].iter().map(|&b| b as i8 as f32).collect(),
        }
    }

    fn to_i32(&self, x: &Storage) -> Vec<i32> {
        let s = st(x);
        match s.dtype {
            DType::I32 => self.read_words::<i32>(s),
            _ => self.to_f32(x).into_iter().map(|v| v as i32).collect(),
        }
    }

    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage {
        let out = self.alloc(len, dtype);
        if dtype == DType::I8 {
            let b = (value as i8) as u8 as u32;
            self.each("fill_i8", &[u(words(len, dtype)), b], &[&out], words(len, dtype));
            return wrap(out);
        }
        let bits = match dtype {
            DType::I32 => (value as i32) as u32,
            DType::F16 => half::f16::from_f32(value).to_f32().to_bits(),
            DType::BF16 => half::bf16::from_f32(value).to_f32().to_bits(),
            _ => value.to_bits(),
        };
        self.each("fill", &[u(len), bits], &[&out], len);
        wrap(out)
    }

    fn cast(&self, x: &Storage, dtype: DType) -> Storage {
        wrap(self.convert(st(x), dtype))
    }

    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
        let s = st(x);
        let xf = self.as_f32(s);
        let out = self.alloc(s.len, DType::F32);
        self.each("unary", &[u(s.len), unary_code(op)], &[&xf, &out], s.len);
        let dt = if s.dtype.is_float() { s.dtype } else { DType::F32 };
        wrap(self.finish(out, dt))
    }

    fn binary(&self, op: BinaryOp, a: &Storage, a_shape: &[usize], b: &Storage, b_shape: &[usize], out_shape: &[usize]) -> Storage {
        // ints (comparisons) run in f32; otherwise both operands share a's dtype
        let dt = if st(a).dtype.is_float() { st(a).dtype } else { DType::F32 };
        let (af, bf) = (self.as_f32(st(a)), self.as_f32(&self.convert(st(b), dt)));
        let n = numel(out_shape);
        let out = self.alloc(n, DType::F32);
        let mut p = vec![u(n), u(out_shape.len()), binary_code(op), 0];
        p.extend(fill8(out_shape));
        p.extend(fill8(&broadcast_strides(a_shape, out_shape)));
        p.extend(fill8(&broadcast_strides(b_shape, out_shape)));
        self.each("binary", &p, &[&af, &bf, &out], n);
        wrap(self.finish(out, dt))
    }

    fn expand(&self, x: &Storage, shape: &[usize], out_shape: &[usize]) -> Storage {
        wrap(self.strided_copy_raw(st(x), out_shape, &broadcast_strides(shape, out_shape), 0))
    }

    fn where_cond(&self, cond: &Storage, a: &Storage, b: &Storage) -> Storage {
        let a = st(a);
        let c = self.as_f32(st(cond));
        let b = self.convert(st(b), a.dtype);
        if a.dtype == DType::I8 {
            let (a32, b32) = (self.convert(a, DType::I32), self.convert(&b, DType::I32));
            let r = self.where_cond(&wrap(c), &wrap(a32), &wrap(b32));
            return wrap(self.convert(st(&r), DType::I8));
        }
        let out = self.alloc(a.len, a.dtype);
        self.each("where", &[u(a.len)], &[&c, a, &b, &out], a.len);
        wrap(out)
    }

    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let s = st(x);
        let xf = self.as_f32(s);
        let (outer, len, inner) = split_axis(shape, axis);
        let code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
            ReduceOp::ArgMax => 3,
        };
        let out = self.alloc(outer * inner, if op == ReduceOp::ArgMax { DType::I32 } else { DType::F32 });
        let p = [u(outer), u(len), u(inner), code];
        if inner == 1 && op != ReduceOp::ArgMax {
            self.run("reduce_rows", &p, &[&xf, &out], rows_grid(outer));
        } else {
            self.each("reduce_strided", &p, &[&xf, &out], outer * inner);
        }
        if op == ReduceOp::ArgMax {
            return wrap(out);
        }
        let dt = if s.dtype.is_float() { s.dtype } else { DType::F32 };
        wrap(self.finish(out, dt))
    }

    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
        let MatmulSpec { batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride } = spec;
        let dt = st(a).dtype;
        if k == 0 {
            return wrap(self.finish(self.zeros(batch * m * n), dt));
        }
        let (af, bf) = (self.as_f32(st(a)), self.as_f32(st(b)));
        let out = self.alloc(batch * m * n, DType::F32);
        let p = [u(batch), u(m), u(k), u(n), trans_a as u32, trans_b as u32, u(a_batch_stride), u(b_batch_stride)];
        // a few rows (decoding): matrix x vector; large matrices: 128 x 128
        // tiles; small ones: 64 x 64 (more workgroups)
        if m <= 8 && batch == 1 {
            let pv = [u(m), u(k), u(n), trans_a as u32, trans_b as u32];
            self.run("gemv", &pv, &[&af, &bf, &out], rows_grid(m * n.div_ceil(32)));
        } else if m >= 256 && n >= 256 {
            self.run("matmul", &p, &[&af, &bf, &out], (n.div_ceil(128) as u32, m.div_ceil(128) as u32, u(batch)));
        } else {
            self.run("matmul64", &p, &[&af, &bf, &out], (n.div_ceil(64) as u32, m.div_ceil(64) as u32, u(batch)));
        }
        wrap(self.finish(out, dt))
    }

    fn write_slice(&self, dst: Storage, shape: &[usize], axis: usize, start: usize, src: &Storage) -> Storage {
        let (_, len, inner) = split_axis(shape, axis);
        let s = st(src);
        let outer_inner = numel(shape) / len.max(1);
        let t = s.len / outer_inner.max(1);
        assert!(start + t <= len, "write_slice: {start}+{t} out of range for size {len}");
        let d = st(&dst);
        assert_eq!(d.dtype, s.dtype, "write_slice: dtype mismatch");
        if d.dtype == DType::I8 {
            let (d32, s32) = (self.convert(d, DType::I32), self.convert(s, DType::I32));
            drop(dst);
            let r = self.write_slice(wrap(d32), shape, axis, start, &wrap(s32));
            return wrap(self.convert(st(&r), DType::I8));
        }
        // in place when nobody else holds the buffer, else into a copy
        let out = if Arc::strong_count(&d.buf) == 1 { d.clone() } else { self.strided_copy_raw(d, &[d.len], &[1], 0) };
        drop(dst);
        self.each("cat_copy", &[u(s.len), u(t * inner), u(len * inner), u(start * inner)], &[s, &out], s.len);
        wrap(out)
    }

    fn strided_copy(&self, x: &Storage, shape: &[usize], strides: &[usize], offset: usize) -> Storage {
        assert!(shape.len() <= MAX_RANK, "GPU tensors have at most 8 dims, got {shape:?}");
        wrap(self.strided_copy_raw(st(x), shape, strides, offset))
    }

    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage {
        let st_in = strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let in_st: Vec<usize> = perm.iter().map(|&p| st_in[p]).collect();
        wrap(self.strided_copy_raw(st(x), &out_shape, &in_st, 0))
    }

    fn narrow(&self, x: &Storage, shape: &[usize], axis: usize, start: usize, len: usize) -> Storage {
        let (outer, l, inner) = split_axis(shape, axis);
        wrap(self.strided_copy_raw(st(x), &[outer, len, inner], &[l * inner, inner, 1], start * inner))
    }

    fn cat(&self, xs: &[&Storage], shapes: &[&[usize]], axis: usize) -> Storage {
        let dtype = xs[0].dtype();
        if dtype == DType::I8 {
            let wide: Vec<Storage> = xs.iter().map(|x| wrap(self.convert(st(x), DType::I32))).collect();
            let refs: Vec<&Storage> = wide.iter().collect();
            let r = self.cat(&refs, shapes, axis);
            return wrap(self.convert(st(&r), DType::I8));
        }
        let (_, _, inner) = split_axis(shapes[0], axis);
        let total: usize = xs.iter().map(|s| s.len()).sum();
        let blk_out: usize = shapes.iter().map(|s| s[axis]).sum::<usize>() * inner;
        let out = self.alloc(total, dtype);
        let mut offset = 0;
        for (x, s) in xs.iter().zip(shapes) {
            let blk_in = s[axis] * inner;
            let x = st(x);
            self.each("cat_copy", &[u(x.len), u(blk_in), u(blk_out), u(offset)], &[x, &out], x.len);
            offset += blk_in;
        }
        wrap(out)
    }

    fn index_select(&self, table: &Storage, cols: usize, idx: &Storage) -> Storage {
        let (t, idx) = (st(table), st(idx));
        if t.dtype == DType::I8 {
            let r = self.index_select(&wrap(self.convert(t, DType::I32)), cols, &wrap(idx.clone()));
            return wrap(self.convert(st(&r), DType::I8));
        }
        let n = idx.len * cols;
        let out = self.alloc(n, t.dtype);
        self.each("index_select", &[u(n), u(cols)], &[t, idx, &out], n);
        wrap(out)
    }

    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &Storage) -> Storage {
        let s = st(src);
        let out = self.zeros(rows * cols);
        let sf = self.as_f32(s);
        self.each("index_add", &[u(s.len), u(cols)], &[&sf, st(idx), &out], s.len);
        wrap(self.finish(out, if s.dtype.is_float() { s.dtype } else { DType::F32 }))
    }

    fn gather(&self, x: &Storage, shape: &[usize], axis: usize, idx: &Storage, idx_len: usize) -> Storage {
        let (s, idx) = (st(x), st(idx));
        let xf = self.as_f32(s);
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.alloc(idx.len, DType::F32);
        self.each("gather", &[u(idx.len), u(len), u(idx_len), u(inner)], &[&xf, idx, &out], idx.len);
        wrap(self.finish(out, if s.dtype.is_half() { s.dtype } else { DType::F32 }))
    }

    fn scatter_add(&self, src: &Storage, idx: &Storage, idx_len: usize, shape: &[usize], axis: usize) -> Storage {
        let (s, idx) = (st(src), st(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let out = self.zeros(numel(shape));
        let sf = self.as_f32(s);
        self.each("scatter_add", &[u(idx.len), u(len), u(idx_len), u(inner)], &[&sf, idx, &out], idx.len);
        wrap(self.finish(out, if s.dtype.is_half() { s.dtype } else { DType::F32 }))
    }

    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
        let (oh, ow) = g.out_hw();
        let n = g.n * g.c * g.kh * g.kw * oh * ow;
        let s = st(x);
        let out = self.alloc(n, s.dtype);
        self.each("im2col", &conv_params(g), &[s, &out], n);
        wrap(out)
    }

    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage {
        let n = g.n * g.c * g.h * g.w;
        let s = st(cols);
        let out = self.alloc(n, DType::F32);
        self.each("col2im", &conv_params(g), &[&self.as_f32(s), &out], n);
        wrap(self.finish(out, s.dtype))
    }

    // ---- fused ops (last axis); other axes use the composed default ----

    fn softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).softmax(x, shape, axis);
        }
        let s = st(x);
        wrap(self.finish(self.softmax_rows(&self.as_f32(s), shape, false), s.dtype))
    }

    fn log_softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).log_softmax(x, shape, axis);
        }
        let s = st(x);
        wrap(self.finish(self.softmax_rows(&self.as_f32(s), shape, true), s.dtype))
    }

    fn softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).softmax_backward(y, g, shape, axis);
        }
        let s = st(y);
        wrap(self.finish(self.softmax_backward_rows(&self.as_f32(s), &self.as_f32(st(g)), shape, false), s.dtype))
    }

    fn log_softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return super::reference::Unfused(backend()).log_softmax_backward(y, g, shape, axis);
        }
        let s = st(y);
        wrap(self.finish(self.softmax_backward_rows(&self.as_f32(s), &self.as_f32(st(g)), shape, true), s.dtype))
    }

    fn layer_norm(&self, x: &Storage, rows: usize, dim: usize, eps: f32) -> (Storage, Storage) {
        let s = st(x);
        let xhat = self.alloc(rows * dim, DType::F32);
        let rstd = self.alloc(rows, DType::F32);
        self.run("layer_norm_rows", &[u(rows), u(dim), eps.to_bits()], &[&self.as_f32(s), &xhat, &rstd], rows_grid(rows));
        (wrap(self.finish(xhat, s.dtype)), wrap(self.finish(rstd, s.dtype)))
    }

    fn layer_norm_backward(&self, g: &Storage, xhat: &Storage, rstd: &Storage, rows: usize, dim: usize) -> Storage {
        let s = st(g);
        let out = self.alloc(rows * dim, DType::F32);
        let (g32, x32, r32) = (self.as_f32(s), self.as_f32(st(xhat)), self.as_f32(st(rstd)));
        self.run("layer_norm_backward_rows", &[u(rows), u(dim)], &[&g32, &x32, &r32, &out], rows_grid(rows));
        wrap(self.finish(out, s.dtype))
    }

    fn matmul_int8(&self, x: &Storage, q: &Storage, scales: &Storage, m: usize, k: usize, n: usize) -> Storage {
        let dt = st(x).dtype;
        if m > 16 {
            // many rows: dequantize once and use the GEMM
            let w = self.cast(q, dt);
            let w = self.binary(BinaryOp::Mul, &w, &[k, n], &self.cast(scales, dt), &[n], &[k, n]);
            let spec = MatmulSpec { batch: 1, m, k, n, trans_a: false, trans_b: false, a_batch_stride: 0, b_batch_stride: 0 };
            return self.matmul(x, &w, spec);
        }
        let out = self.alloc(m * n, DType::F32);
        let tiles = m * n.div_ceil(32);
        self.run("matmul_int8", &[u(m), u(k), u(n)], &[&self.as_f32(st(x)), st(q), &self.as_f32(st(scales)), &out], rows_grid(tiles));
        wrap(self.finish(out, dt))
    }

    fn dequant_int4(&self, q: &Storage, scales: &Storage, k: usize, n: usize, group: usize) -> Storage {
        let out = self.alloc(k * n, DType::F32);
        self.each("dequant_int4", &[0, u(k), u(n), u(group)], &[st(q), &self.as_f32(st(scales)), &out], k * n);
        wrap(out)
    }

    fn matmul_int4(&self, x: &Storage, q: &Storage, scales: &Storage, m: usize, k: usize, n: usize, group: usize) -> Storage {
        let dt = st(x).dtype;
        if m > 16 {
            // many rows: dequantize once and use the GEMM
            let w = self.cast(&self.dequant_int4(q, scales, k, n, group), dt);
            let spec = MatmulSpec { batch: 1, m, k, n, trans_a: false, trans_b: false, a_batch_stride: 0, b_batch_stride: 0 };
            return self.matmul(x, &w, spec);
        }
        let out = self.alloc(m * n, DType::F32);
        let tiles = m * n.div_ceil(32);
        self.run("matmul_int4", &[u(m), u(k), u(n), u(group)], &[&self.as_f32(st(x)), st(q), &self.as_f32(st(scales)), &out], rows_grid(tiles));
        wrap(self.finish(out, dt))
    }

    fn adam_step(&self, p: &Storage, g: &Storage, m: &Storage, v: &Storage, h: AdamParams) -> (Storage, Storage, Storage) {
        let n = st(p).len;
        let (np, nm, nv) = (self.alloc(n, DType::F32), self.alloc(n, DType::F32), self.alloc(n, DType::F32));
        let params = [
            u(n),
            h.lr.to_bits(),
            h.beta1.to_bits(),
            h.beta2.to_bits(),
            h.eps.to_bits(),
            h.weight_decay.to_bits(),
            h.decoupled as u32,
            (1.0 / h.bias_correction1).to_bits(),
            (1.0 / h.bias_correction2).to_bits(),
        ];
        let (pf, gf, mf, vf) = (self.as_f32(st(p)), self.as_f32(st(g)), self.as_f32(st(m)), self.as_f32(st(v)));
        self.each("adam_step", &params, &[&pf, &gf, &mf, &vf, &np, &nm, &nv], n);
        (wrap(np), wrap(nm), wrap(nv))
    }
}
