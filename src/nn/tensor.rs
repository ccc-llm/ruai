//! The `Tensor` type and the reverse-mode autograd engine.

use crate::nn::backend::{backend_for, Backend, DType, Storage};
use crate::nn::device::Device;
use crate::nn::rand;
use crate::nn::shape::numel;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Computes the gradients of a node's parents from the gradient of its output.
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

pub(crate) struct Node {
    parents: Vec<Tensor>,
    backward: BackwardFn,
    name: &'static str,
}

/// The data behind a tensor: a buffer plus how the tensor's elements are
/// laid out in it (a strided view; contiguous tensors have row-major strides
/// and offset 0).
struct Data {
    storage: Storage,
    /// Element strides per dim (0 for broadcast dims).
    strides: Vec<usize>,
    /// Element offset of the first element.
    offset: usize,
    /// Cached contiguous copy of a non-contiguous view (data is immutable).
    contiguous: Option<Storage>,
}

/// Row-major strides of `shape`.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

/// Whether `strides` are row-major for `shape` (ignoring size-1 dims).
pub(crate) fn is_row_major(shape: &[usize], strides: &[usize]) -> bool {
    let mut expect = 1;
    for d in (0..shape.len()).rev() {
        if shape[d] != 1 {
            if strides[d] != expect {
                return false;
            }
            expect *= shape[d];
        }
    }
    true
}

impl Data {
    fn new(storage: Storage, shape: &[usize]) -> Self {
        Data { storage, strides: contiguous_strides(shape), offset: 0, contiguous: None }
    }

    /// The storage holds exactly this tensor's elements in row-major order.
    fn is_plain(&self, shape: &[usize]) -> bool {
        self.offset == 0 && self.storage.len() == numel(shape) && is_row_major(shape, &self.strides)
    }
}

/// Shared state behind a [`Tensor`] (storage, shape, gradient, graph node).
pub struct TensorImpl {
    id: usize,
    data: RwLock<Data>,
    shape: Vec<usize>,
    requires_grad: AtomicBool,
    grad: RwLock<Option<Tensor>>,
    node: Option<Node>,
}

/// An n-dimensional array with optional gradient tracking. Elements are one
/// [`DType`] (`F32`, `F16`, `BF16`, `I32` or `I8`) and live on one [`Device`]
/// (CPU, or MPS / CUDA with the matching cargo feature).
/// Cloning a `Tensor` is cheap: clones share the same underlying data.
#[derive(Clone)]
pub struct Tensor(Arc<TensorImpl>);

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
    static AUTOCAST: Cell<Option<DType>> = const { Cell::new(None) };
}

/// The half-precision dtype of the active [`autocast`] region, if any.
pub fn autocast_dtype() -> Option<DType> {
    AUTOCAST.with(|a| a.get())
}

/// RAII guard for an autocast region (see [`autocast`]); `None` disables it.
pub struct AutocastGuard {
    prev: Option<DType>,
}

impl AutocastGuard {
    /// Enter an autocast region with `dtype` (`F16`/`BF16`, panics otherwise) or disable autocast with `None`.
    pub fn new(dtype: Option<DType>) -> Self {
        if let Some(d) = dtype {
            assert!(d.is_half(), "autocast dtype must be F16 or BF16, got {d}");
        }
        AutocastGuard { prev: AUTOCAST.with(|a| a.replace(dtype)) }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|a| a.set(self.prev));
    }
}

/// Run `f` with automatic mixed precision (like `torch.autocast`):
/// matmul / linear / conv run in `dtype` (F16 or BF16) while softmax,
/// layer/batch norm and losses run in f32. Parameters stay f32 ("master
/// weights") and receive f32 gradients. Pair with
/// [`crate::nn::amp::GradScaler`] when using F16.
pub fn autocast<T>(dtype: DType, f: impl FnOnce() -> T) -> T {
    let _g = AutocastGuard::new(Some(dtype));
    f()
}

/// Whether operations are currently recorded for autograd (false inside [`no_grad`]).
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// RAII guard that disables graph construction while alive (like `torch.no_grad()`).
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    /// Disable grad mode until the guard is dropped.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let prev = GRAD_ENABLED.with(|g| g.replace(false));
        NoGradGuard { prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

/// Run `f` with gradient recording switched on, even inside [`no_grad`] or a
/// backward pass (like `torch.enable_grad()`).
pub fn enable_grad<T>(f: impl FnOnce() -> T) -> T {
    let prev = GRAD_ENABLED.with(|g| g.replace(true));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GRAD_ENABLED.with(|g| g.set(self.0));
        }
    }
    let _r = Restore(prev);
    f()
}

/// Run `f` without recording operations for autograd.
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _g = NoGradGuard::new();
    f()
}

impl Tensor {
    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    fn new_impl(storage: Storage, shape: Vec<usize>, requires_grad: bool, node: Option<Node>) -> Tensor {
        assert_eq!(storage.len(), numel(&shape), "storage length does not match shape {shape:?}");
        Tensor(Arc::new(TensorImpl {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(Data::new(storage, &shape)),
            shape,
            requires_grad: AtomicBool::new(requires_grad),
            grad: RwLock::new(None),
            node,
        }))
    }

    /// Wrap a storage as a leaf tensor (no gradient).
    pub fn from_storage(storage: Storage, shape: Vec<usize>) -> Tensor {
        Self::new_impl(storage, shape, false, None)
    }

    /// Create the result of a differentiable operation. The graph node is only
    /// recorded if grad mode is on and some parent requires grad.
    pub(crate) fn from_op<F>(
        storage: Storage,
        shape: Vec<usize>,
        parents: Vec<Tensor>,
        name: &'static str,
        backward: F,
    ) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static,
    {
        if is_grad_enabled() && parents.iter().any(|p| p.requires_grad()) {
            let node = Node { parents, backward: Box::new(backward), name };
            Self::new_impl(storage, shape, true, Some(node))
        } else {
            Self::from_storage(storage, shape)
        }
    }

    /// A view of `base`'s buffer with its own shape / strides / offset (no
    /// copy). Recorded for autograd like [`Tensor::from_op`] with `base` as parent.
    pub(crate) fn view_op<F>(base: &Tensor, shape: Vec<usize>, strides: Vec<usize>, offset: usize, name: &'static str, backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync + 'static,
    {
        let storage = base.0.data.read().unwrap().storage.clone();
        if numel(&shape) > 0 {
            let last = offset + shape.iter().zip(&strides).map(|(&n, &s)| (n - 1) * s).sum::<usize>();
            assert!(last < storage.len(), "view out of bounds: shape {shape:?}, strides {strides:?}, offset {offset}");
        }
        let record = is_grad_enabled() && base.requires_grad();
        let node = record.then(|| Node { parents: vec![base.clone()], backward: Box::new(backward), name });
        Tensor(Arc::new(TensorImpl {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(Data { storage, strides, offset, contiguous: None }),
            shape,
            requires_grad: AtomicBool::new(record),
            grad: RwLock::new(None),
            node,
        }))
    }

    /// A leaf tensor viewing `storage` with explicit element `strides` and
    /// `offset` (no copy; panics if the view does not fit in the buffer).
    pub fn from_storage_strided(storage: Storage, shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Tensor {
        assert_eq!(shape.len(), strides.len(), "from_storage_strided: shape and strides differ in rank");
        if numel(&shape) > 0 {
            let last = offset + shape.iter().zip(&strides).map(|(&n, &s)| (n - 1) * s).sum::<usize>();
            assert!(last < storage.len(), "view out of bounds: shape {shape:?}, strides {strides:?}, offset {offset}");
        }
        Tensor(Arc::new(TensorImpl {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(Data { storage, strides, offset, contiguous: None }),
            shape,
            requires_grad: AtomicBool::new(false),
            grad: RwLock::new(None),
            node: None,
        }))
    }

    /// An `F32` tensor from row-major data (panics if `data.len()` doesn't match `shape`).
    pub fn from_vec(data: Vec<f32>, shape: &[usize], device: Device) -> Tensor {
        assert_eq!(data.len(), numel(shape), "data length {} != numel of {shape:?}", data.len());
        Self::from_storage(backend_for(device).from_f32(data), shape.to_vec())
    }

    /// An `I32` tensor (e.g. token ids, class targets).
    pub fn from_vec_i32(data: Vec<i32>, shape: &[usize], device: Device) -> Tensor {
        assert_eq!(data.len(), numel(shape), "data length {} != numel of {shape:?}", data.len());
        Self::from_storage(backend_for(device).from_i32(data), shape.to_vec())
    }

    /// An `I8` tensor (e.g. quantized weights).
    pub fn from_vec_i8(data: Vec<i8>, shape: &[usize], device: Device) -> Tensor {
        assert_eq!(data.len(), numel(shape), "data length {} != numel of {shape:?}", data.len());
        Self::from_storage(backend_for(device).from_i8(data), shape.to_vec())
    }

    /// An `I32` index tensor from `usize` indices.
    pub fn from_indices(idx: &[usize], shape: &[usize], device: Device) -> Tensor {
        Self::from_vec_i32(idx.iter().map(|&i| i as i32).collect(), shape, device)
    }

    /// An `F32` CPU tensor copied from `data`.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Tensor {
        Self::from_vec(data.to_vec(), shape, Device::Cpu)
    }

    /// A rank-0 tensor.
    pub fn scalar(v: f32, device: Device) -> Tensor {
        Self::from_vec(vec![v], &[], device)
    }

    /// An `F32` tensor filled with `value`.
    pub fn full(shape: &[usize], value: f32, device: Device) -> Tensor {
        Self::from_storage(backend_for(device).full(numel(shape), value, DType::F32), shape.to_vec())
    }

    /// An `F32` tensor of zeros.
    pub fn zeros(shape: &[usize], device: Device) -> Tensor {
        Self::full(shape, 0.0, device)
    }

    /// An `F32` tensor of ones.
    pub fn ones(shape: &[usize], device: Device) -> Tensor {
        Self::full(shape, 1.0, device)
    }

    /// Zeros with `t`'s shape, device and dtype.
    pub fn zeros_like(t: &Tensor) -> Tensor {
        Self::full_like(t, 0.0)
    }

    /// Ones with `t`'s shape, device and dtype.
    pub fn ones_like(t: &Tensor) -> Tensor {
        Self::full_like(t, 1.0)
    }

    /// `value` everywhere, with `t`'s shape, device and dtype.
    pub fn full_like(t: &Tensor, value: f32) -> Tensor {
        Self::from_storage(t.backend().full(t.numel(), value, t.dtype()), t.shape().to_vec())
    }

    /// Samples from N(0, 1).
    pub fn randn(shape: &[usize], device: Device) -> Tensor {
        Self::from_vec(rand::normal_vec(numel(shape), 0.0, 1.0), shape, device)
    }

    /// Samples from U[low, high).
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32, device: Device) -> Tensor {
        Self::from_vec(rand::uniform_vec(numel(shape), low, high), shape, device)
    }

    /// `[0, 1, ..., n-1]`
    pub fn arange(n: usize, device: Device) -> Tensor {
        Self::from_vec((0..n).map(|i| i as f32).collect(), &[n], device)
    }

    /// Enable gradient tracking (builder style), e.g. `Tensor::randn(..).requires_grad_()`.
    pub fn requires_grad_(self) -> Tensor {
        self.set_requires_grad(true);
        self
    }

    /// Turn gradient tracking on or off. Panics for non-float tensors, or when disabling it on a non-leaf.
    pub fn set_requires_grad(&self, v: bool) {
        assert!(!v || self.dtype().is_float(), "only floating-point tensors can require grad");
        assert!(self.0.node.is_none() || v, "cannot disable grad on a non-leaf tensor");
        self.0.requires_grad.store(v, Ordering::Relaxed);
    }

    // ------------------------------------------------------------------
    // accessors
    // ------------------------------------------------------------------

    /// Unique id of this tensor (shared by its clones).
    pub fn id(&self) -> usize {
        self.0.id
    }

    /// Dimensions, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        numel(&self.0.shape)
    }

    /// Size of dimension `d` (negative indexes from the end).
    pub fn dim(&self, d: isize) -> usize {
        self.0.shape[crate::nn::shape::norm_dim(d, self.rank())]
    }

    /// The underlying storage (a cheap reference-counted clone).
    pub fn storage(&self) -> Storage {
        {
            let d = self.0.data.read().unwrap();
            if d.is_plain(&self.0.shape) {
                return d.storage.clone();
            }
            if let Some(c) = &d.contiguous {
                return c.clone();
            }
        }
        // materialize the view once (the data never changes)
        let mut d = self.0.data.write().unwrap();
        if let Some(c) = &d.contiguous {
            return c.clone();
        }
        let c = backend_for(d.storage.device()).strided_copy(&d.storage, &self.0.shape, &d.strides, d.offset);
        d.contiguous = Some(c.clone());
        c
    }

    /// Whether the tensor is a plain row-major buffer (not a view needing a copy).
    pub fn is_contiguous(&self) -> bool {
        self.0.data.read().unwrap().is_plain(&self.0.shape)
    }

    /// The underlying buffer with this tensor's element `(strides, offset)`
    /// in it, without copying (for view-aware kernels).
    pub fn strided_parts(&self) -> (Storage, Vec<usize>, usize) {
        let d = self.0.data.read().unwrap();
        (d.storage.clone(), d.strides.clone(), d.offset)
    }

    /// Device the data lives on.
    pub fn device(&self) -> Device {
        self.0.data.read().unwrap().storage.device()
    }

    pub(crate) fn backend(&self) -> &'static dyn Backend {
        backend_for(self.device())
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.0.data.read().unwrap().storage.dtype()
    }

    /// Convert to another dtype. Float-to-float casts are differentiable
    /// (the gradient is cast back); casts to or from integers are not.
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        let src = self.dtype();
        if src == dtype {
            return self.clone();
        }
        let s = self.backend().cast(&self.storage(), dtype);
        if src.is_float() && dtype.is_float() {
            Tensor::from_op(s, self.shape().to_vec(), vec![self.clone()], "to_dtype", move |g| {
                vec![Some(g.to_dtype(src))]
            })
        } else {
            Tensor::from_storage(s, self.shape().to_vec())
        }
    }

    /// Whether gradients are tracked for this tensor.
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    /// `true` if not produced by a recorded op (e.g. a parameter or input).
    pub fn is_leaf(&self) -> bool {
        self.0.node.is_none()
    }

    /// Name of the op that produced this tensor (for debugging).
    pub fn op_name(&self) -> Option<&'static str> {
        self.0.node.as_ref().map(|n| n.name)
    }

    /// Copy the data to the host as `f32` (integer tensors are converted).
    pub fn to_vec(&self) -> Vec<f32> {
        self.storage().to_vec()
    }

    /// Copy the data to the host as `i32`.
    pub fn to_vec_i32(&self) -> Vec<i32> {
        self.storage().to_vec_i32()
    }

    /// Value of a single-element tensor.
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() needs a single-element tensor, got {:?}", self.shape());
        self.to_vec()[0]
    }

    /// Accumulated gradient (after `backward()`).
    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.read().unwrap().clone()
    }

    /// Overwrite the accumulated gradient.
    pub fn set_grad(&self, g: Option<Tensor>) {
        *self.0.grad.write().unwrap() = g;
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        self.set_grad(None);
    }

    /// Replace the data in place (used by optimizers / loading weights / moving
    /// modules between devices). The shape must not change.
    pub fn set_storage(&self, s: Storage) {
        assert_eq!(s.len(), self.numel(), "set_storage: size mismatch");
        assert_eq!(s.dtype(), self.dtype(), "set_storage: dtype mismatch");
        *self.0.data.write().unwrap() = Data::new(s, &self.0.shape);
    }

    /// Copy the values of `src` into this tensor in place (keeps this tensor's device).
    pub fn copy_(&self, src: &Tensor) {
        assert_eq!(self.shape(), src.shape(), "copy_: shape mismatch");
        self.set_storage(src.storage().to_device(self.device()));
    }

    /// Replace the data in place, allowing a dtype change (same number of
    /// elements); used by [`crate::nn::modules::Module::to_dtype`].
    pub fn set_data(&self, s: Storage) {
        assert_eq!(s.len(), self.numel(), "set_data: size mismatch");
        *self.0.data.write().unwrap() = Data::new(s, &self.0.shape);
        self.zero_grad();
    }

    /// A new leaf tensor sharing the same data, cut from the graph.
    pub fn detach(&self) -> Tensor {
        let d = self.0.data.read().unwrap();
        Tensor(Arc::new(TensorImpl {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(Data { storage: d.storage.clone(), strides: d.strides.clone(), offset: d.offset, contiguous: d.contiguous.clone() }),
            shape: self.0.shape.clone(),
            requires_grad: AtomicBool::new(false),
            grad: RwLock::new(None),
            node: None,
        }))
    }

    // ------------------------------------------------------------------
    // autograd
    // ------------------------------------------------------------------

    /// Reverse-mode differentiation from this tensor. For non-scalar outputs
    /// the seed gradient is all ones (i.e. this computes d(sum(self))/dx).
    pub fn backward(&self) {
        self.backward_with(Tensor::ones_like(self));
    }

    /// Reverse-mode differentiation with an explicit seed gradient (same shape as `self`); leaf gradients accumulate into `.grad()`.
    pub fn backward_with(&self, seed: Tensor) {
        assert!(self.requires_grad(), "backward() on a tensor that does not require grad");
        run_backward(&[(self.clone(), seed)], false, None);
    }

    /// Like [`Tensor::backward`], but the accumulated `.grad` tensors carry
    /// their own autograd graph, so they can be differentiated again (second
    /// derivatives, gradient penalties). Like `backward(create_graph=True)`.
    pub fn backward_create_graph(&self) {
        assert!(self.requires_grad(), "backward() on a tensor that does not require grad");
        run_backward(&[(self.clone(), Tensor::ones_like(self))], true, None);
    }
}

/// The reverse-mode engine. Propagates `seed` gradients from each root to
/// everything it depends on, in reverse topological order.
///
/// * `create_graph`: run the backward functions with gradient recording on,
///   so the resulting gradients are differentiable themselves.
/// * `capture`: return the total gradient of each of these tensors (leaves
///   or not) instead of accumulating into the leaves' `.grad`.
pub(crate) fn run_backward(roots: &[(Tensor, Tensor)], create_graph: bool, capture: Option<&[Tensor]>) -> Vec<Option<Tensor>> {
    for (r, seed) in roots {
        assert_eq!(seed.shape(), r.shape(), "seed gradient shape mismatch");
    }
    let _grad_mode = (!create_graph).then(NoGradGuard::new);
    let _enabled = create_graph.then(|| {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                GRAD_ENABLED.with(|g| g.set(self.0));
            }
        }
        Restore(GRAD_ENABLED.with(|g| g.replace(true)))
    });
    let _no_autocast = AutocastGuard::new(None);

    // Topological order (parents before children) via iterative DFS.
    let mut order: Vec<Tensor> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Tensor, bool)> = roots.iter().filter(|(r, _)| r.requires_grad()).map(|(r, _)| (r.clone(), false)).collect();
    while let Some((t, expanded)) = stack.pop() {
        if expanded {
            order.push(t);
            continue;
        }
        if !visited.insert(t.id()) {
            continue;
        }
        stack.push((t.clone(), true));
        if let Some(node) = &t.0.node {
            for p in &node.parents {
                if p.requires_grad() && !visited.contains(&p.id()) {
                    stack.push((p.clone(), false));
                }
            }
        }
    }

    let targets: HashMap<usize, usize> = capture.unwrap_or(&[]).iter().enumerate().map(|(i, t)| (t.id(), i)).collect();
    let mut captured: Vec<Option<Tensor>> = vec![None; targets.len()];
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    for (r, seed) in roots {
        let acc = match grads.remove(&r.id()) {
            Some(old) => &old + seed,
            None => seed.clone(),
        };
        grads.insert(r.id(), acc);
    }
    for t in order.iter().rev() {
        let Some(g) = grads.remove(&t.id()) else { continue };
        if let Some(&i) = targets.get(&t.id()) {
            captured[i] = Some(g.clone());
        }
        match &t.0.node {
            None => {
                if capture.is_none() {
                    // leaf: accumulate into .grad
                    let new = match t.grad() {
                        Some(old) => &old + &g,
                        None => g,
                    };
                    t.set_grad(Some(new));
                }
            }
            Some(node) => {
                let pgrads = (node.backward)(&g);
                debug_assert_eq!(pgrads.len(), node.parents.len(), "{}: wrong number of grads", node.name);
                for (p, pg) in node.parents.iter().zip(pgrads) {
                    let Some(mut pg) = pg else { continue };
                    if !p.requires_grad() {
                        continue;
                    }
                    if pg.dtype() != p.dtype() {
                        pg = pg.to_dtype(p.dtype());
                    }
                    assert_eq!(
                        pg.shape(),
                        p.shape(),
                        "{}: gradient shape {:?} != input shape {:?}",
                        node.name,
                        pg.shape(),
                        p.shape()
                    );
                    let acc = match grads.remove(&p.id()) {
                        Some(old) => &old + &pg,
                        None => pg,
                    };
                    grads.insert(p.id(), acc);
                }
            }
        }
    }
    captured
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.to_vec();
        let shown: Vec<String> = v.iter().take(8).map(|x| format!("{x:.4}")).collect();
        let more = if v.len() > 8 { ", ..." } else { "" };
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={}, data=[{}{}]{})",
            self.shape(),
            self.dtype(),
            self.device(),
            shown.join(", "),
            more,
            if self.requires_grad() { ", requires_grad" } else { "" }
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
