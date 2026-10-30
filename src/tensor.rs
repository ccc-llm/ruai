//! The `Tensor` type and the reverse-mode autograd engine.

use crate::backend::{backend_for, Backend, DType, Storage};
use crate::device::Device;
use crate::rand;
use crate::shape::numel;
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

pub struct TensorImpl {
    id: usize,
    storage: RwLock<Storage>,
    shape: Vec<usize>,
    requires_grad: AtomicBool,
    grad: RwLock<Option<Tensor>>,
    node: Option<Node>,
}

/// An n-dimensional `f32` array with optional gradient tracking.
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
/// [`crate::amp::GradScaler`] when using F16.
pub fn autocast<T>(dtype: DType, f: impl FnOnce() -> T) -> T {
    let _g = AutocastGuard::new(Some(dtype));
    f()
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// RAII guard that disables graph construction while alive (like `torch.no_grad()`).
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
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
            storage: RwLock::new(storage),
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

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Tensor {
        Self::from_vec(data.to_vec(), shape, Device::Cpu)
    }

    /// A rank-0 tensor.
    pub fn scalar(v: f32, device: Device) -> Tensor {
        Self::from_vec(vec![v], &[], device)
    }

    pub fn full(shape: &[usize], value: f32, device: Device) -> Tensor {
        Self::from_storage(backend_for(device).full(numel(shape), value, DType::F32), shape.to_vec())
    }

    pub fn zeros(shape: &[usize], device: Device) -> Tensor {
        Self::full(shape, 0.0, device)
    }

    pub fn ones(shape: &[usize], device: Device) -> Tensor {
        Self::full(shape, 1.0, device)
    }

    pub fn zeros_like(t: &Tensor) -> Tensor {
        Self::zeros(t.shape(), t.device())
    }

    pub fn ones_like(t: &Tensor) -> Tensor {
        Self::ones(t.shape(), t.device())
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

    pub fn set_requires_grad(&self, v: bool) {
        assert!(!v || self.dtype().is_float(), "only floating-point tensors can require grad");
        assert!(self.0.node.is_none() || v, "cannot disable grad on a non-leaf tensor");
        self.0.requires_grad.store(v, Ordering::Relaxed);
    }

    // ------------------------------------------------------------------
    // accessors
    // ------------------------------------------------------------------

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }

    pub fn numel(&self) -> usize {
        numel(&self.0.shape)
    }

    /// Size of dimension `d` (negative indexes from the end).
    pub fn dim(&self, d: isize) -> usize {
        self.0.shape[crate::shape::norm_dim(d, self.rank())]
    }

    pub fn storage(&self) -> Storage {
        self.0.storage.read().unwrap().clone()
    }

    pub fn device(&self) -> Device {
        self.0.storage.read().unwrap().device()
    }

    pub(crate) fn backend(&self) -> &'static dyn Backend {
        backend_for(self.device())
    }

    pub fn dtype(&self) -> DType {
        self.0.storage.read().unwrap().dtype()
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

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

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

    pub fn set_grad(&self, g: Option<Tensor>) {
        *self.0.grad.write().unwrap() = g;
    }

    pub fn zero_grad(&self) {
        self.set_grad(None);
    }

    /// Replace the data in place (used by optimizers / loading weights / moving
    /// modules between devices). The shape must not change.
    pub fn set_storage(&self, s: Storage) {
        assert_eq!(s.len(), self.numel(), "set_storage: size mismatch");
        assert_eq!(s.dtype(), self.dtype(), "set_storage: dtype mismatch");
        *self.0.storage.write().unwrap() = s;
    }

    /// Copy the values of `src` into this tensor in place (keeps this tensor's device).
    pub fn copy_(&self, src: &Tensor) {
        assert_eq!(self.shape(), src.shape(), "copy_: shape mismatch");
        self.set_storage(src.storage().to_device(self.device()));
    }

    /// Replace the data in place, allowing a dtype change (same number of
    /// elements); used by [`crate::nn::Module::to_dtype`].
    pub fn set_data(&self, s: Storage) {
        assert_eq!(s.len(), self.numel(), "set_data: size mismatch");
        *self.0.storage.write().unwrap() = s;
        self.zero_grad();
    }

    /// A new leaf tensor sharing the same data, cut from the graph.
    pub fn detach(&self) -> Tensor {
        Tensor::from_storage(self.storage(), self.shape().to_vec())
    }

    // ------------------------------------------------------------------
    // autograd
    // ------------------------------------------------------------------

    /// Reverse-mode differentiation from this tensor. For non-scalar outputs
    /// the seed gradient is all ones (i.e. this computes d(sum(self))/dx).
    pub fn backward(&self) {
        self.backward_with(Tensor::ones_like(self));
    }

    pub fn backward_with(&self, seed: Tensor) {
        assert!(self.requires_grad(), "backward() on a tensor that does not require grad");
        assert_eq!(seed.shape(), self.shape(), "seed gradient shape mismatch");
        let _guard = NoGradGuard::new();
        let _no_autocast = AutocastGuard::new(None);

        // Topological order (parents before children) via iterative DFS.
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(Tensor, bool)> = vec![(self.clone(), false)];
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

        let mut grads: HashMap<usize, Tensor> = HashMap::new();
        grads.insert(self.id(), seed);
        for t in order.iter().rev() {
            let Some(g) = grads.remove(&t.id()) else { continue };
            match &t.0.node {
                None => {
                    // leaf: accumulate into .grad
                    let new = match t.grad() {
                        Some(old) => &old + &g,
                        None => g,
                    };
                    t.set_grad(Some(new));
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
    }
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
