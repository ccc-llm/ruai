//! `torch` for py0: a subset of PyTorch on `ruai::nn` (float32 tensors with
//! autograd), declared by `src/py/lib/torch*.pyi`. It follows PyTorch where it
//! shows: the CPU generator (MT19937, so `torch.manual_seed` gives PyTorch's
//! `rand`, `randn` and `nn.Linear` initial weights), dtypes (float32, int64,
//! bool), `grad_fn` names, printing (`tensor([...], grad_fn=<AddBackward0>)`),
//! module reprs and the optimizers' update formulas. Matrix products sum in
//! order, so results agree with PyTorch's BLAS to float32 rounding.

mod format;
pub mod nn;

/// `torch.optim`
pub mod optim {
    pub use super::nn::{adam, sgd, Optimizer};
}

use crate::nn::{Device, Tensor as T};
use crate::pyrt::err::{PyErr, INDEX_ERROR, TYPE_ERROR, VALUE_ERROR};
use std::cell::RefCell;

pub(crate) const RUNTIME_ERROR: &[&str] = &["RuntimeError", "Exception", "BaseException"];

pub(crate) fn rerr(msg: impl Into<String>) -> PyErr {
    PyErr::new(RUNTIME_ERROR, msg.into())
}

/// The element type a tensor shows (values are stored as float32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    Bool,
    Int64,
    Float32,
}

impl DType {
    fn name(self) -> &'static str {
        match self {
            DType::Bool => "torch.bool",
            DType::Int64 => "torch.int64",
            DType::Float32 => "torch.float32",
        }
    }
    fn parse(s: &str) -> Result<DType, PyErr> {
        Ok(match s {
            "float32" | "float" | "torch.float32" => DType::Float32,
            "int64" | "long" | "torch.int64" => DType::Int64,
            "bool" | "torch.bool" => DType::Bool,
            _ => return Err(PyErr::new(TYPE_ERROR, format!("ruai: dtype {:?} is not supported (float32, int64, bool)", s))),
        })
    }
}

// ------------------------------------------------------------ the generator

/// PyTorch's CPU generator: MT19937 with a cached second normal sample.
struct Gen {
    mt: crate::rng::Mt19937,
    normal: Option<f64>,
}

thread_local! {
    static GEN: RefCell<Gen> = RefCell::new(Gen { mt: crate::rng::Mt19937::new(67280421310721u64 as u32), normal: None });
}

fn with_gen<R>(f: impl FnOnce(&mut Gen) -> R) -> R {
    GEN.with(|g| f(&mut g.borrow_mut()))
}

impl Gen {
    /// `uniform_real_distribution<float>`: 24 random bits.
    fn uniform_f32(&mut self) -> f32 {
        ((self.mt.next_u32() & 0xff_ffff) as f64 * (1.0 / (1u64 << 24) as f64)) as f32
    }
    /// `uniform_real_distribution<double>`: 53 bits of a 64-bit draw.
    fn uniform_f64(&mut self) -> f64 {
        let hi = self.mt.next_u32() as u64;
        let lo = self.mt.next_u32() as u64;
        (((hi << 32) | lo) & ((1u64 << 53) - 1)) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// `normal_distribution<double>`: Box-Muller, the second value cached.
    fn normal_f64(&mut self) -> f64 {
        if let Some(v) = self.normal.take() {
            return v;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.normal = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// `torch.manual_seed(seed)`
pub fn manual_seed(seed: i64) {
    with_gen(|g| {
        g.mt = crate::rng::Mt19937::new(seed as u64 as u32);
        g.normal = None;
    });
}

/// `n` values uniform in `[from, to)` (`tensor.uniform_`, computed in double).
pub(crate) fn uniform(n: usize, from: f64, to: f64) -> Vec<f32> {
    with_gen(|g| {
        (0..n)
            .map(|_| {
                let x = (g.mt.next_u32() & 0xff_ffff) as f64 * (1.0 / (1u64 << 24) as f64);
                (x * (to - from) + from) as f32
            })
            .collect()
    })
}

/// `n` standard normal values (`torch.randn`).
fn normal(n: usize) -> Vec<f32> {
    with_gen(|g| {
        if n < 16 {
            return (0..n).map(|_| g.normal_f64() as f32).collect();
        }
        // normal_fill: uniforms first, then Box-Muller on blocks of 16 (float)
        let mut d: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
        let fill16 = |d: &mut [f32]| {
            for j in 0..8 {
                let u1 = 1.0f32 - d[j];
                let u2 = d[j + 8];
                let radius = (-2.0f32 * u1.ln()).sqrt();
                let theta = 2.0f32 * std::f32::consts::PI * u2;
                d[j] = radius * theta.cos();
                d[j + 8] = radius * theta.sin();
            }
        };
        let mut i = 0;
        while i + 16 <= n {
            fill16(&mut d[i..i + 16]);
            i += 16;
        }
        if n % 16 != 0 {
            let tail = n - 16;
            for x in d[tail..].iter_mut() {
                *x = g.uniform_f32();
            }
            fill16(&mut d[tail..]);
        }
        d
    })
}

// ------------------------------------------------------------ tensors

/// `torch.Tensor`
#[derive(Clone)]
pub struct Tensor {
    pub(crate) t: T,
    pub(crate) dt: DType,
    pub(crate) grad_fn: Option<&'static str>,
    /// an `nn.Parameter` (printed "Parameter containing:")
    pub(crate) param: bool,
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor::new(T::from_vec(vec![], &[0], Device::Cpu), DType::Float32)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, o: &Tensor) -> bool {
        self.t.id() == o.t.id()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.__repr__())
    }
}

/// `torch.Size`
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Size(pub Vec<i64>);

impl Size {
    /// `str(size)`: `torch.Size([2, 3])`
    pub fn __str__(&self) -> String {
        format!("torch.Size([{}])", self.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
    /// `repr(size)`
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
    /// `size[i]`
    pub fn __getitem__(&self, i: i64) -> Result<i64, PyErr> {
        let n = self.0.len() as i64;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(PyErr::new(INDEX_ERROR, "tuple index out of range"));
        }
        Ok(self.0[j as usize])
    }
    /// `len(size)`
    pub fn __len__(&self) -> i64 {
        self.0.len() as i64
    }
    /// `size == other`
    pub fn __eq__(&self, o: &Size) -> bool {
        self.0 == o.0
    }
    /// The dimensions as a list.
    pub fn tolist(&self) -> Vec<i64> {
        self.0.clone()
    }
    /// `size.numel()`
    pub fn numel(&self) -> i64 {
        self.0.iter().product()
    }
}

/// An operand: a tensor or a Python scalar.
pub enum Arg {
    T(Tensor),
    S(f64, DType),
}

/// Values the operators take: tensors and Python bool / int / float.
pub trait Operand {
    fn arg(&self) -> Arg;
}
impl Operand for f64 {
    fn arg(&self) -> Arg {
        Arg::S(*self, DType::Float32)
    }
}
impl Operand for i64 {
    fn arg(&self) -> Arg {
        Arg::S(*self as f64, DType::Int64)
    }
}
impl Operand for bool {
    fn arg(&self) -> Arg {
        Arg::S(*self as i64 as f64, DType::Bool)
    }
}
impl Operand for Tensor {
    fn arg(&self) -> Arg {
        Arg::T(self.clone())
    }
}
impl<X: Operand + ?Sized> Operand for &X {
    fn arg(&self) -> Arg {
        (**self).arg()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, PyErr> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return Err(rerr(format!(
                "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension {}",
                x,
                y,
                i
            )));
        };
    }
    Ok(out)
}

fn int_like(d: DType) -> bool {
    d != DType::Float32
}

impl Tensor {
    pub(crate) fn new(t: T, dt: DType) -> Tensor {
        Tensor { t, dt, grad_fn: None, param: false }
    }
    /// A result: `grad_fn` when it is part of a graph.
    pub(crate) fn op(t: T, dt: DType, name: &'static str) -> Tensor {
        let g = if t.requires_grad() { Some(name) } else { None };
        Tensor { t, dt, grad_fn: g, param: false }
    }
    pub(crate) fn floats(v: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::new(T::from_vec(v, shape, Device::Cpu), DType::Float32)
    }
    fn scalar_t(v: f64) -> T {
        T::scalar(v as f32, Device::Cpu)
    }
    pub(crate) fn dims(&self) -> Vec<usize> {
        self.t.shape().to_vec()
    }
    fn values(&self) -> Vec<f32> {
        self.t.to_vec()
    }
    fn float_only(&self, what: &str) -> Result<(), PyErr> {
        if self.dt != DType::Float32 {
            return Err(rerr(format!(
                "{}: could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {}",
                what,
                if self.dt == DType::Int64 { "Long" } else { "Bool" }
            )));
        }
        Ok(())
    }

    fn binary(&self, o: Arg, op: Op, reflected: bool) -> Result<Tensor, PyErr> {
        let (ot, odt, scalar) = match o {
            Arg::T(t) => (t.t.clone(), t.dt, false),
            Arg::S(v, d) => (Tensor::scalar_t(v), d, true),
        };
        let (a, b) = if reflected { (ot.clone(), self.t.clone()) } else { (self.t.clone(), ot.clone()) };
        broadcast(a.shape(), b.shape())?;
        // the result type: a Python scalar does not widen an int64 / bool tensor to float unless it is a float
        let dt = match (self.dt, odt) {
            (x, y) if x == DType::Float32 || y == DType::Float32 => DType::Float32,
            (DType::Bool, DType::Bool) if !scalar => DType::Bool,
            _ => DType::Int64,
        };
        let cmp = |t: T| -> Result<Tensor, PyErr> { Ok(Tensor::new(t, DType::Bool)) };
        let elementwise = |a: &T, b: &T, f: &dyn Fn(f32, f32) -> f32| -> Result<T, PyErr> {
            let shape = broadcast(a.shape(), b.shape())?;
            let (x, y) = (a.expand(&shape).contiguous().to_vec(), b.expand(&shape).contiguous().to_vec());
            Ok(T::from_vec(x.iter().zip(&y).map(|(p, q)| f(*p, *q)).collect(), &shape, Device::Cpu))
        };
        match op {
            Op::Add => Ok(Tensor::op(&a + &b, dt, "AddBackward0")),
            Op::Sub => {
                if self.dt == DType::Bool && odt == DType::Bool {
                    return Err(rerr("Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead."));
                }
                Ok(Tensor::op(&a - &b, dt, if reflected && scalar { "RsubBackward1" } else { "SubBackward0" }))
            }
            Op::Mul => Ok(Tensor::op(&a * &b, dt, "MulBackward0")),
            Op::Div => {
                let name = if reflected && scalar { "MulBackward0" } else { "DivBackward0" };
                Ok(Tensor::op(&a / &b, DType::Float32, name))
            }
            Op::Pow => {
                if !reflected && scalar {
                    let p = ot.item();
                    let r = if p == 2.0 { &a * &a } else { a.powf(p) };
                    if int_like(dt) {
                        if p < 0.0 {
                            return Err(rerr("Integers to negative integer powers are not allowed."));
                        }
                        return Ok(Tensor::new(elementwise(&self.t, &ot, &|x, y| x.powf(y))?, dt));
                    }
                    return Ok(Tensor::op(r, dt, "PowBackward0"));
                }
                Err(rerr("ruai: only tensor ** scalar is supported"))
            }
            Op::Lt => cmp(elementwise(&a, &b, &|x, y| (x < y) as i64 as f32)?),
            Op::Le => cmp(elementwise(&a, &b, &|x, y| (x <= y) as i64 as f32)?),
            Op::Gt => cmp(elementwise(&a, &b, &|x, y| (x > y) as i64 as f32)?),
            Op::Ge => cmp(elementwise(&a, &b, &|x, y| (x >= y) as i64 as f32)?),
            Op::Eq => cmp(elementwise(&a, &b, &|x, y| (x == y) as i64 as f32)?),
            Op::Ne => cmp(elementwise(&a, &b, &|x, y| (x != y) as i64 as f32)?),
        }
    }
}

macro_rules! ops {
    ($($name:ident $rname:ident $op:ident),* $(,)?) => {
        impl Tensor {
            $(
                #[doc = concat!("`a ", stringify!($op), " b`")]
                pub fn $name(&self, o: impl Operand) -> Result<Tensor, PyErr> {
                    self.binary(o.arg(), Op::$op, false)
                }
                #[doc = concat!("`b ", stringify!($op), " a`")]
                pub fn $rname(&self, o: impl Operand) -> Result<Tensor, PyErr> {
                    self.binary(o.arg(), Op::$op, true)
                }
            )*
        }
    };
}
ops! { __add__ __radd__ Add, __sub__ __rsub__ Sub, __mul__ __rmul__ Mul, __truediv__ __rtruediv__ Div, __pow__ __rpow__ Pow }

macro_rules! cmps {
    ($($name:ident $op:ident),*) => {
        impl Tensor {
            $(
                #[doc = concat!("`a ", stringify!($op), " b`: a bool tensor")]
                pub fn $name(&self, o: impl Operand) -> Result<Tensor, PyErr> {
                    self.binary(o.arg(), Op::$op, false)
                }
            )*
        }
    };
}
cmps! { __lt__ Lt, __le__ Le, __gt__ Gt, __ge__ Ge, __eq__ Eq, __ne__ Ne }

fn dim_index(d: i64, n: usize) -> Result<isize, PyErr> {
    let m = n.max(1) as i64;
    if d < -m || d >= m {
        return Err(PyErr::new(
            INDEX_ERROR,
            format!("Dimension out of range (expected to be in range of [{}, {}], but got {})", -m, m - 1, d),
        ));
    }
    Ok(if d < 0 { d + m } else { d } as isize)
}

impl Tensor {
    // ------------------------------------------------------------ attributes

    /// `t.shape`
    pub fn shape(&self) -> Size {
        Size(self.dims().iter().map(|&d| d as i64).collect())
    }
    /// `t.size()`
    pub fn size(&self) -> Size {
        self.shape()
    }
    /// `t.size(dim)`
    pub fn size_dim(&self, dim: i64) -> Result<i64, PyErr> {
        let d = dim_index(dim, self.dims().len())?;
        Ok(self.dims()[d as usize] as i64)
    }
    /// `t.dim()`
    pub fn dim(&self) -> i64 {
        self.dims().len() as i64
    }
    /// `t.numel()`
    pub fn numel(&self) -> i64 {
        self.dims().iter().product::<usize>() as i64
    }
    /// `t.dtype`: `torch.float32`, `torch.int64`, `torch.bool`
    pub fn dtype(&self) -> String {
        self.dt.name().to_string()
    }
    /// `t.requires_grad`
    pub fn requires_grad(&self) -> bool {
        self.t.requires_grad()
    }
    /// `t.is_leaf`
    pub fn is_leaf(&self) -> bool {
        self.t.is_leaf()
    }
    /// `t.grad`
    pub fn grad(&self) -> Option<Tensor> {
        self.t.grad().map(|g| Tensor::new(g, DType::Float32))
    }
    /// `t.requires_grad_(True)`
    pub fn requires_grad_(&self, on: bool) -> Result<Tensor, PyErr> {
        if on && self.dt != DType::Float32 {
            return Err(rerr("only Tensors of floating point dtype can require gradients"));
        }
        if !self.t.is_leaf() {
            return Err(rerr("you can only change requires_grad flags of leaf variables."));
        }
        self.t.set_requires_grad(on);
        Ok(self.clone())
    }
    /// `t.detach()`
    pub fn detach(&self) -> Tensor {
        Tensor::new(self.t.detach(), self.dt)
    }
    /// `t.clone()`
    pub fn clone_(&self) -> Tensor {
        let r = Tensor::new(self.t.mul_scalar(1.0), self.dt);
        Tensor { grad_fn: if r.t.requires_grad() { Some("CloneBackward0") } else { None }, ..r }
    }
    /// `t.item()`
    pub fn item(&self) -> Result<f64, PyErr> {
        if self.numel() != 1 {
            return Err(rerr(format!("a Tensor with {} elements cannot be converted to Scalar", self.numel())));
        }
        Ok(self.t.item() as f64)
    }
    /// `float(t)`
    pub fn __float__(&self) -> Result<f64, PyErr> {
        self.item()
    }
    /// `int(t)`
    pub fn __int__(&self) -> Result<i64, PyErr> {
        Ok(self.item()?.trunc() as i64)
    }
    /// `bool(t)`, `if t:`
    pub fn __bool__(&self) -> Result<bool, PyErr> {
        if self.numel() != 1 {
            return Err(rerr("Boolean value of Tensor with more than one value is ambiguous"));
        }
        Ok(self.t.item() != 0.0)
    }
    /// `len(t)`
    pub fn __len__(&self) -> Result<i64, PyErr> {
        match self.dims().first() {
            Some(&d) => Ok(d as i64),
            None => Err(PyErr::new(TYPE_ERROR, "len() of a 0-d tensor")),
        }
    }
    /// `t.tolist()` of a 1-d tensor, as floats.
    pub fn tolist(&self) -> Result<Vec<f64>, PyErr> {
        if self.dims().len() != 1 {
            return Err(PyErr::new(VALUE_ERROR, "ruai: tolist() is for 1-d tensors (a list of floats)"));
        }
        Ok(self.values().into_iter().map(|x| x as f64).collect())
    }
    /// `t.numpy()`
    pub fn numpy(&self) -> Result<super::numpy::NdArray, PyErr> {
        if self.t.requires_grad() {
            return Err(rerr("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."));
        }
        let shape: Vec<i64> = self.dims().iter().map(|&d| d as i64).collect();
        let v: Vec<f64> = self.values().into_iter().map(|x| x as f64).collect();
        let k = match self.dt {
            DType::Float32 => "float32",
            DType::Int64 => "int64",
            DType::Bool => "bool",
        };
        super::numpy::array(v, &Some("float64".into()))?.astype(k)?.reshape(shape)
    }
    /// `str(t)`
    pub fn __str__(&self) -> String {
        format::tensor_repr(self)
    }
    /// `repr(t)`
    pub fn __repr__(&self) -> String {
        format::tensor_repr(self)
    }

    // ------------------------------------------------------------ autograd

    /// `t.backward()`
    pub fn backward(&self) -> Result<(), PyErr> {
        if !self.t.requires_grad() {
            return Err(rerr("element 0 of tensors does not require grad and does not have a grad_fn"));
        }
        if self.numel() != 1 {
            return Err(rerr("grad can be implicitly created only for scalar outputs"));
        }
        self.t.backward();
        Ok(())
    }

    // ------------------------------------------------------------ math

    /// `-t`
    pub fn __neg__(&self) -> Result<Tensor, PyErr> {
        if self.dt == DType::Bool {
            return Err(rerr("Negation, the `-` operator, on a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."));
        }
        Ok(Tensor::op(self.t.neg(), self.dt, "NegBackward0"))
    }
    /// `a @ b`
    pub fn __matmul__(&self, o: &Tensor) -> Result<Tensor, PyErr> {
        let (a, b) = (self.dims(), o.dims());
        let dt = self.dt.max(o.dt);
        let mismatch = |x: String, y: String| rerr(format!("mat1 and mat2 shapes cannot be multiplied ({} and {})", x, y));
        match (a.len(), b.len()) {
            (1, 1) => {
                if a[0] != b[0] {
                    return Err(rerr(format!(
                        "inconsistent tensor size, expected tensor [{}] and src [{}] to have the same number of elements, but got {} and {} elements respectively",
                        a[0], b[0], a[0], b[0]
                    )));
                }
                let r = (&self.t * &o.t).sum_all();
                Ok(Tensor::op(r, dt, "DotBackward0"))
            }
            (2, 2) => {
                if a[1] != b[0] {
                    return Err(mismatch(format!("{}x{}", a[0], a[1]), format!("{}x{}", b[0], b[1])));
                }
                Ok(Tensor::op(self.t.matmul(&o.t), dt, "MmBackward0"))
            }
            (2, 1) => {
                if a[1] != b[0] {
                    return Err(rerr(format!("size mismatch, got input ({}), mat ({}x{}), vec ({})", a[0], a[0], a[1], b[0])));
                }
                Ok(Tensor::op(self.t.matmul(&o.t.unsqueeze(1)).squeeze(1), dt, "MvBackward0"))
            }
            (1, 2) => {
                if a[0] != b[0] {
                    return Err(mismatch(format!("1x{}", a[0]), format!("{}x{}", b[0], b[1])));
                }
                Ok(Tensor::op(self.t.unsqueeze(0).matmul(&o.t).squeeze(0), dt, "SqueezeBackward4"))
            }
            _ => Err(rerr("ruai: matmul of tensors with more than 2 dimensions is not supported")),
        }
    }
    fn unary(&self, t: T, name: &'static str) -> Tensor {
        Tensor::op(t, DType::Float32, name)
    }
    /// `t.relu()`
    pub fn relu(&self) -> Tensor {
        Tensor::op(self.t.relu(), self.dt, "ReluBackward0")
    }
    /// `t.sigmoid()`
    pub fn sigmoid(&self) -> Tensor {
        self.unary(self.t.sigmoid(), "SigmoidBackward0")
    }
    /// `t.tanh()`
    pub fn tanh(&self) -> Tensor {
        self.unary(self.t.tanh(), "TanhBackward0")
    }
    /// `t.exp()`
    pub fn exp(&self) -> Tensor {
        self.unary(self.t.exp(), "ExpBackward0")
    }
    /// `t.log()`
    pub fn log(&self) -> Tensor {
        self.unary(self.t.log(), "LogBackward0")
    }
    /// `t.sqrt()`
    pub fn sqrt(&self) -> Tensor {
        self.unary(self.t.sqrt(), "SqrtBackward0")
    }
    /// `t.abs()`
    pub fn abs(&self) -> Tensor {
        Tensor::op(self.t.abs(), self.dt, "AbsBackward0")
    }
    /// `abs(t)`
    pub fn __abs__(&self) -> Tensor {
        self.abs()
    }
    /// `t.softmax(dim)`
    pub fn softmax(&self, dim: i64) -> Result<Tensor, PyErr> {
        let d = dim_index(dim, self.dims().len())?;
        Ok(self.unary(self.t.softmax(d), "SoftmaxBackward0"))
    }
    /// `t.log_softmax(dim)`
    pub fn log_softmax(&self, dim: i64) -> Result<Tensor, PyErr> {
        let d = dim_index(dim, self.dims().len())?;
        Ok(self.unary(self.t.log_softmax(d), "LogSoftmaxBackward0"))
    }

    // ------------------------------------------------------------ reductions

    fn sum_dt(&self) -> DType {
        if self.dt == DType::Float32 {
            DType::Float32
        } else {
            DType::Int64
        }
    }
    /// `t.sum(dim=None, keepdim=False)`
    pub fn sum(&self, dim: Option<i64>, keepdim: bool) -> Result<Tensor, PyErr> {
        match dim {
            None => Ok(Tensor::op(self.t.sum_all(), self.sum_dt(), "SumBackward0")),
            Some(d) => {
                let d = dim_index(d, self.dims().len())?;
                Ok(Tensor::op(self.t.sum(d, keepdim), self.sum_dt(), "SumBackward1"))
            }
        }
    }
    /// `t.mean(dim=None, keepdim=False)`
    pub fn mean(&self, dim: Option<i64>, keepdim: bool) -> Result<Tensor, PyErr> {
        self.float_only("mean()")?;
        match dim {
            None => Ok(Tensor::op(self.t.mean_all(), DType::Float32, "MeanBackward0")),
            Some(d) => {
                let d = dim_index(d, self.dims().len())?;
                Ok(Tensor::op(self.t.mean(d, keepdim), DType::Float32, "MeanBackward1"))
            }
        }
    }
    /// `t.max()`: the largest element
    pub fn max(&self) -> Result<Tensor, PyErr> {
        if self.numel() == 0 {
            return Err(rerr("max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument."));
        }
        let flat = self.t.reshape(&[self.numel() as usize]);
        Ok(Tensor::op(flat.max(0, false), self.dt, "MaxBackward1"))
    }
    /// `t.max(dim)`: (values, indices)
    pub fn max_dim(&self, dim: i64, keepdim: bool) -> Result<(Tensor, Tensor), PyErr> {
        let d = dim_index(dim, self.dims().len())?;
        let v = Tensor::op(self.t.max(d, keepdim), self.dt, "MaxBackward0");
        let i = Tensor::new(self.t.detach().argmax(d, keepdim).to_dtype(crate::nn::DType::F32), DType::Int64);
        Ok((v, i))
    }
    /// `t.argmax(dim=None)`
    pub fn argmax(&self, dim: Option<i64>) -> Result<Tensor, PyErr> {
        let t = self.t.detach();
        let r = match dim {
            None => t.reshape(&[self.numel() as usize]).argmax(0, false),
            Some(d) => t.argmax(dim_index(d, self.dims().len())?, false),
        };
        Ok(Tensor::new(r.to_dtype(crate::nn::DType::F32), DType::Int64))
    }

    // ------------------------------------------------------------ shapes

    fn new_shape(&self, want: &[i64]) -> Result<Vec<usize>, PyErr> {
        let n = self.numel();
        let known: i64 = want.iter().filter(|&&d| d != -1).product();
        let unknown = want.iter().filter(|&&d| d == -1).count();
        if unknown > 1 {
            return Err(rerr("only one dimension can be inferred"));
        }
        let dims: Vec<i64> = want.iter().map(|&d| if d == -1 && known != 0 { n / known } else { d }).collect();
        if dims.iter().product::<i64>() != n || dims.iter().any(|&d| d < 0) {
            return Err(rerr(format!(
                "shape '[{}]' is invalid for input of size {}",
                want.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
                n
            )));
        }
        Ok(dims.into_iter().map(|d| d as usize).collect())
    }
    /// `t.view(shape)` / `t.reshape(shape)`
    pub fn view(&self, shape: &[i64]) -> Result<Tensor, PyErr> {
        let s = self.new_shape(shape)?;
        Ok(Tensor::op(self.t.reshape(&s), self.dt, "ViewBackward0"))
    }
    /// `t.flatten()`
    pub fn flatten(&self) -> Result<Tensor, PyErr> {
        self.view(&[-1])
    }
    /// `t.t()`
    pub fn t(&self) -> Result<Tensor, PyErr> {
        match self.dims().len() {
            0 | 1 => Ok(self.clone()),
            2 => Ok(Tensor::op(self.t.t(), self.dt, "TBackward0")),
            n => Err(rerr(format!("t() expects a tensor with <= 2 dimensions, but self is {}D", n))),
        }
    }
    /// `t.T`
    #[allow(non_snake_case)]
    pub fn T(&self) -> Tensor {
        let n = self.dims().len();
        if n < 2 {
            return self.clone();
        }
        let mut t = self.t.clone();
        for i in 0..n / 2 {
            t = t.transpose(i as isize, (n - 1 - i) as isize);
        }
        Tensor::op(t, self.dt, "PermuteBackward0")
    }
    /// `t.unsqueeze(dim)`
    pub fn unsqueeze(&self, dim: i64) -> Result<Tensor, PyErr> {
        let d = dim_index(dim, self.dims().len() + 1)?;
        Ok(Tensor::op(self.t.unsqueeze(d), self.dt, "UnsqueezeBackward0"))
    }
    /// `t.squeeze(dim)`
    pub fn squeeze(&self, dim: i64) -> Result<Tensor, PyErr> {
        let d = dim_index(dim, self.dims().len())?;
        if self.dims()[d as usize] != 1 {
            return Ok(Tensor::op(self.t.clone(), self.dt, "SqueezeBackward1"));
        }
        Ok(Tensor::op(self.t.squeeze(d), self.dt, "SqueezeBackward1"))
    }
    /// `t.round()`: half to even
    pub fn round(&self) -> Tensor {
        let v: Vec<f32> = self.values().into_iter().map(|x| x.round_ties_even()).collect();
        let r = T::from_vec(v, &self.dims(), Device::Cpu);
        Tensor { grad_fn: if self.t.requires_grad() { Some("RoundBackward0") } else { None }, ..Tensor::new(r, self.dt) }
    }
    /// `t.float()`
    pub fn float(&self) -> Tensor {
        Tensor { dt: DType::Float32, ..self.clone() }
    }
    /// `t.long()`
    pub fn long(&self) -> Tensor {
        let v: Vec<f32> = self.values().into_iter().map(|x| x.trunc()).collect();
        Tensor::new(T::from_vec(v, &self.dims(), Device::Cpu), DType::Int64)
    }

    // ------------------------------------------------------------ indexing

    /// `t[i]`, `t[a:b]`, `t[i, j]`, `t[:, j]`
    pub fn __getitem__(&self, index: &crate::pyrt::Index) -> Result<Tensor, PyErr> {
        use crate::pyrt::Ix;
        let mut t = self.t.clone();
        let mut dim = 0isize;
        let mut name = "SelectBackward0";
        for part in &index.0 {
            let shape = t.shape().to_vec();
            if dim as usize >= shape.len() {
                return Err(PyErr::new(
                    INDEX_ERROR,
                    format!("too many indices for tensor of dimension {}", self.dims().len()),
                ));
            }
            let n = shape[dim as usize] as i64;
            match part {
                Ix::Int(i) => {
                    let j = if *i < 0 { i + n } else { *i };
                    if j < 0 || j >= n {
                        return Err(PyErr::new(
                            INDEX_ERROR,
                            format!("index {} is out of bounds for dimension {} with size {}", i, dim, n),
                        ));
                    }
                    t = t.select(dim, j as usize);
                    name = "SelectBackward0";
                }
                Ix::Slice(a, b, step) => {
                    if step.is_some_and(|s| s != 1) {
                        return Err(PyErr::new(VALUE_ERROR, "ruai: tensor slices with a step are not supported"));
                    }
                    let fix = |x: Option<i64>, d: i64| match x {
                        None => d,
                        Some(x) if x < 0 => (x + n).max(0),
                        Some(x) => x.min(n),
                    };
                    let (s, e) = (fix(*a, 0), fix(*b, n));
                    let len = (e - s).max(0) as usize;
                    t = t.narrow(dim, s as usize, len);
                    name = "SliceBackward0";
                    dim += 1;
                }
                _ => return Err(PyErr::new(INDEX_ERROR, "ruai: tensors are indexed by ints and slices")),
            }
        }
        Ok(Tensor::op(t, self.dt, name))
    }
}

// ------------------------------------------------------------ creation

/// Python values that make a tensor: (nested) lists of bool, int or float, or a number.
pub trait IntoTensor {
    fn items(&self, shape: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, depth: usize) -> Result<(), PyErr>;
}

macro_rules! leaf {
    ($t:ty, $d:expr) => {
        impl IntoTensor for $t {
            fn items(&self, _s: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, _d: usize) -> Result<(), PyErr> {
                out.push((*self as i64 as f64, $d));
                Ok(())
            }
        }
    };
}
leaf!(bool, DType::Bool);
leaf!(i64, DType::Int64);
impl IntoTensor for f64 {
    fn items(&self, _s: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, _d: usize) -> Result<(), PyErr> {
        out.push((*self, DType::Float32));
        Ok(())
    }
}
impl<X: IntoTensor> IntoTensor for [X] {
    fn items(&self, shape: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, depth: usize) -> Result<(), PyErr> {
        if shape.len() == depth {
            shape.push(self.len());
        } else if shape[depth] != self.len() {
            return Err(PyErr::new(
                VALUE_ERROR,
                format!("expected sequence of length {} at dim {} (got {})", shape[depth], depth, self.len()),
            ));
        }
        for x in self {
            x.items(shape, out, depth + 1)?;
        }
        Ok(())
    }
}
impl<X: IntoTensor> IntoTensor for Vec<X> {
    fn items(&self, shape: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, depth: usize) -> Result<(), PyErr> {
        self.as_slice().items(shape, out, depth)
    }
}
impl<X: IntoTensor + ?Sized> IntoTensor for &X {
    fn items(&self, shape: &mut Vec<usize>, out: &mut Vec<(f64, DType)>, depth: usize) -> Result<(), PyErr> {
        (**self).items(shape, out, depth)
    }
}

/// `torch.tensor(data, dtype=None, requires_grad=False)`
pub fn tensor(data: impl IntoTensor, dtype: &Option<String>, requires_grad: bool) -> Result<Tensor, PyErr> {
    let mut shape = vec![];
    let mut items = vec![];
    data.items(&mut shape, &mut items, 0)?;
    let dt = match dtype {
        Some(d) => DType::parse(d)?,
        None => items.iter().map(|x| x.1).max().unwrap_or(DType::Float32),
    };
    let v: Vec<f32> = items
        .iter()
        .map(|&(x, _)| match dt {
            DType::Float32 => x as f32,
            DType::Int64 => x.trunc() as f32,
            DType::Bool => (x != 0.0) as i64 as f32,
        })
        .collect();
    let t = Tensor::new(T::from_vec(v, &shape, Device::Cpu), dt);
    if requires_grad {
        t.requires_grad_(true)?;
    }
    Ok(t)
}

/// A shape argument: `3`, `(2, 3)`, `[2, 3]`.
pub trait SizeArg {
    fn dims(&self) -> Vec<i64>;
}
impl SizeArg for i64 {
    fn dims(&self) -> Vec<i64> {
        vec![*self]
    }
}
impl SizeArg for (i64,) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0]
    }
}
impl SizeArg for (i64, i64) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0, self.1]
    }
}
impl SizeArg for (i64, i64, i64) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0, self.1, self.2]
    }
}
impl SizeArg for [i64] {
    fn dims(&self) -> Vec<i64> {
        self.to_vec()
    }
}
impl SizeArg for Vec<i64> {
    fn dims(&self) -> Vec<i64> {
        self.clone()
    }
}
impl<X: SizeArg + ?Sized> SizeArg for &X {
    fn dims(&self) -> Vec<i64> {
        (**self).dims()
    }
}

fn shape_of(s: &impl SizeArg) -> Result<Vec<usize>, PyErr> {
    s.dims()
        .into_iter()
        .map(|d| if d < 0 { Err(rerr(format!("Trying to create tensor with negative dimension {}: {:?}", d, s.dims()))) } else { Ok(d as usize) })
        .collect()
}

fn filled(shape: Vec<usize>, v: f32, dtype: &Option<String>, requires_grad: bool) -> Result<Tensor, PyErr> {
    let dt = dtype.as_deref().map(DType::parse).transpose()?.unwrap_or(DType::Float32);
    let n = shape.iter().product();
    let t = Tensor::new(T::from_vec(vec![v; n], &shape, Device::Cpu), dt);
    if requires_grad {
        t.requires_grad_(true)?;
    }
    Ok(t)
}

/// `torch.zeros(size, dtype=None, requires_grad=False)`
pub fn zeros(size: impl SizeArg, dtype: &Option<String>, requires_grad: bool) -> Result<Tensor, PyErr> {
    filled(shape_of(&size)?, 0.0, dtype, requires_grad)
}
/// `torch.ones(size, dtype=None, requires_grad=False)`
pub fn ones(size: impl SizeArg, dtype: &Option<String>, requires_grad: bool) -> Result<Tensor, PyErr> {
    filled(shape_of(&size)?, 1.0, dtype, requires_grad)
}
/// `torch.full(size, value)`
pub fn full(size: impl SizeArg, value: impl Operand) -> Result<Tensor, PyErr> {
    let (v, d) = match value.arg() {
        Arg::S(v, d) => (v, d),
        Arg::T(t) => (t.item()?, t.dt),
    };
    let t = filled(shape_of(&size)?, v as f32, &None, false)?;
    Ok(Tensor { dt: d, ..t })
}
/// `torch.zeros_like(t)`
pub fn zeros_like(t: &Tensor) -> Tensor {
    Tensor::new(T::from_vec(vec![0.0; t.numel() as usize], &t.dims(), Device::Cpu), t.dt)
}
/// `torch.ones_like(t)`
pub fn ones_like(t: &Tensor) -> Tensor {
    Tensor::new(T::from_vec(vec![1.0; t.numel() as usize], &t.dims(), Device::Cpu), t.dt)
}
/// `torch.rand(size)`
pub fn rand(size: impl SizeArg, requires_grad: bool) -> Result<Tensor, PyErr> {
    let shape = shape_of(&size)?;
    let v = with_gen(|g| (0..shape.iter().product::<usize>()).map(|_| g.uniform_f32()).collect());
    let t = Tensor::floats(v, &shape);
    if requires_grad {
        t.requires_grad_(true)?;
    }
    Ok(t)
}
/// `torch.randn(size)`
pub fn randn(size: impl SizeArg, requires_grad: bool) -> Result<Tensor, PyErr> {
    let shape = shape_of(&size)?;
    let t = Tensor::floats(normal(shape.iter().product()), &shape);
    if requires_grad {
        t.requires_grad_(true)?;
    }
    Ok(t)
}
/// `torch.arange(start, end, step)` (int64 when all are ints)
pub fn arange(start: impl Operand, end: impl Operand, step: impl Operand) -> Result<Tensor, PyErr> {
    let get = |a: Arg| match a {
        Arg::S(v, d) => (v, d),
        Arg::T(t) => (t.t.item() as f64, t.dt),
    };
    let ((a, da), (b, db), (s, ds)) = (get(start.arg()), get(end.arg()), get(step.arg()));
    if s == 0.0 {
        return Err(rerr("step must be nonzero"));
    }
    let n = ((b - a) / s).ceil().max(0.0) as usize;
    let dt = if [da, db, ds].iter().all(|&d| d != DType::Float32) { DType::Int64 } else { DType::Float32 };
    let v = (0..n).map(|i| (a + i as f64 * s) as f32).collect();
    Ok(Tensor::new(T::from_vec(v, &[n], Device::Cpu), dt))
}
/// `torch.arange(end)`
pub fn arange1(end: impl Operand) -> Result<Tensor, PyErr> {
    arange(0i64, end, 1i64)
}
/// `torch.arange(start, end)`
pub fn arange2(start: impl Operand, end: impl Operand) -> Result<Tensor, PyErr> {
    arange(start, end, 1i64)
}
/// `torch.from_numpy(a)` (a copy; float32, int64 or bool arrays)
pub fn from_numpy(a: &super::numpy::NdArray) -> Result<Tensor, PyErr> {
    let dt = match a.dtype().name().as_str() {
        "float32" => DType::Float32,
        "int64" => DType::Int64,
        "bool" => DType::Bool,
        _ => {
            return Err(PyErr::new(TYPE_ERROR, "ruai: torch.from_numpy needs a float32, int64 or bool array (float64 tensors are not supported: use a.astype(\"float32\"))"))
        }
    };
    let shape: Vec<usize> = a.dims().to_vec();
    Ok(Tensor::new(T::from_vec(a.to_f64().into_iter().map(|x| x as f32).collect(), &shape, Device::Cpu), dt))
}

/// `torch.cat(tensors, dim=0)`
pub fn cat(ts: &[Tensor], dim: i64) -> Result<Tensor, PyErr> {
    let Some(first) = ts.first() else {
        return Err(rerr("torch.cat(): expected a non-empty list of Tensors"));
    };
    let d = dim_index(dim, first.dims().len())?;
    let refs: Vec<&T> = ts.iter().map(|t| &t.t).collect();
    let dt = ts.iter().map(|t| t.dt).max().unwrap();
    Ok(Tensor::op(T::cat(&refs, d), dt, "CatBackward0"))
}
/// `torch.stack(tensors, dim=0)`
pub fn stack(ts: &[Tensor], dim: i64) -> Result<Tensor, PyErr> {
    let Some(first) = ts.first() else {
        return Err(rerr("stack expects a non-empty TensorList"));
    };
    let d = dim_index(dim, first.dims().len() + 1)?;
    let refs: Vec<&T> = ts.iter().map(|t| &t.t).collect();
    let dt = ts.iter().map(|t| t.dt).max().unwrap();
    Ok(Tensor::op(T::stack(&refs, d), dt, "StackBackward0"))
}

/// `torch.no_grad()`
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NoGrad;

thread_local! {
    static GRAD_STACK: RefCell<Vec<bool>> = const { RefCell::new(vec![]) };
}

impl NoGrad {
    /// `torch.no_grad()`
    pub fn new() -> NoGrad {
        NoGrad
    }
    /// `with torch.no_grad():` enters
    pub fn __enter__(&self) {
        let prev = crate::nn::set_grad_enabled(false);
        GRAD_STACK.with(|s| s.borrow_mut().push(prev));
    }
    /// leaves
    pub fn __exit__(&self) {
        let prev = GRAD_STACK.with(|s| s.borrow_mut().pop()).unwrap_or(true);
        crate::nn::set_grad_enabled(prev);
    }
}

macro_rules! py_text {
    ($($t:ty),*) => {
        $(
            impl std::fmt::Display for $t {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str(&self.__str__())
                }
            }
            impl crate::pyrt::Py for $t {
                fn fmt_repr(&self, out: &mut String) {
                    out.push_str(&self.__repr__())
                }
            }
        )*
    };
}

py_text!(Tensor, Size);
