//! `torch.nn` (modules, losses, functional) and `torch.optim`.

use super::{rerr, uniform, DType, Tensor};
use crate::nn::{functional as F, Device, Tensor as T};
use crate::pyrt::err::{PyErr, INDEX_ERROR};
use std::cell::RefCell;
use std::rc::Rc;

const ATTRIBUTE_ERROR: &[&str] = &["AttributeError", "Exception", "BaseException"];

enum Layer {
    Linear { w: Tensor, b: Option<Tensor>, i: usize, o: usize },
    ReLU,
    Sigmoid,
    Tanh,
    Softmax(i64),
    Flatten,
    Identity,
    Sequential(Vec<Module>),
}

/// A `torch.nn` module (`nn.Linear(2, 3)`, `nn.ReLU()`, `nn.Sequential(...)`).
#[derive(Clone)]
pub struct Module(Rc<RefCell<Layer>>);

impl Default for Module {
    fn default() -> Module {
        Module::of(Layer::Identity)
    }
}

impl PartialEq for Module {
    fn eq(&self, o: &Module) -> bool {
        Rc::ptr_eq(&self.0, &o.0)
    }
}

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.__repr__())
    }
}

fn param(v: Vec<f32>, shape: &[usize]) -> Tensor {
    let t = T::from_vec(v, shape, Device::Cpu);
    t.set_requires_grad(true);
    Tensor { t, dt: DType::Float32, grad_fn: None, param: true }
}

impl Module {
    fn of(l: Layer) -> Module {
        Module(Rc::new(RefCell::new(l)))
    }
    fn name(&self) -> &'static str {
        match &*self.0.borrow() {
            Layer::Linear { .. } => "Linear",
            Layer::ReLU => "ReLU",
            Layer::Sigmoid => "Sigmoid",
            Layer::Tanh => "Tanh",
            Layer::Softmax(_) => "Softmax",
            Layer::Flatten => "Flatten",
            Layer::Identity => "Identity",
            Layer::Sequential(_) => "Sequential",
        }
    }
    /// `repr(module)`
    pub fn __repr__(&self) -> String {
        match &*self.0.borrow() {
            Layer::Linear { b, i, o, .. } => {
                format!("Linear(in_features={}, out_features={}, bias={})", i, o, if b.is_some() { "True" } else { "False" })
            }
            Layer::Softmax(d) => format!("Softmax(dim={})", d),
            Layer::Flatten => "Flatten(start_dim=1, end_dim=-1)".into(),
            Layer::Sequential(ms) if ms.is_empty() => "Sequential()".into(),
            Layer::Sequential(ms) => {
                let mut s = "Sequential(\n".to_string();
                for (k, m) in ms.iter().enumerate() {
                    let child = m.__repr__().replace('\n', "\n  ");
                    s += &format!("  ({}): {}\n", k, child);
                }
                s + ")"
            }
            _ => format!("{}()", self.name()),
        }
    }
    /// `str(module)`
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
    /// `module(x)`
    pub fn __call__(&self, x: &Tensor) -> Result<Tensor, PyErr> {
        match &*self.0.borrow() {
            Layer::Linear { w, b, i, o } => {
                let d = x.dims();
                let last = *d.last().unwrap_or(&0);
                if d.is_empty() || last != *i {
                    let rows: usize = if d.len() > 1 { d[..d.len() - 1].iter().product() } else { 1 };
                    return Err(rerr(format!("mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})", rows, last, i, o)));
                }
                let xt = if d.len() == 1 { x.t.unsqueeze(0) } else { x.t.clone() };
                let mut y = xt.matmul(&w.t.t());
                if let Some(b) = b {
                    y = &y + &b.t;
                }
                if d.len() == 1 {
                    y = y.squeeze(0);
                }
                let name = match (d.len(), b.is_some()) {
                    (1, _) => "ViewBackward0",
                    (2, true) => "AddmmBackward0",
                    (2, false) => "MmBackward0",
                    _ => "ViewBackward0",
                };
                Ok(Tensor::op(y, DType::Float32, name))
            }
            Layer::ReLU => Ok(x.relu()),
            Layer::Sigmoid => Ok(x.sigmoid()),
            Layer::Tanh => Ok(x.tanh()),
            Layer::Softmax(d) => x.softmax(*d),
            Layer::Identity => Ok(x.clone()),
            Layer::Flatten => {
                let d = x.dims();
                if d.len() < 2 {
                    return Ok(x.clone());
                }
                x.view(&[d[0] as i64, -1])
            }
            Layer::Sequential(ms) => {
                let mut y = x.clone();
                for m in ms {
                    y = m.__call__(&y)?;
                }
                Ok(y)
            }
        }
    }
    /// `module.forward(x)`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, PyErr> {
        self.__call__(x)
    }
    /// `module.parameters()` (as a list)
    pub fn parameters(&self) -> Vec<Tensor> {
        match &*self.0.borrow() {
            Layer::Linear { w, b, .. } => std::iter::once(w.clone()).chain(b.clone()).collect(),
            Layer::Sequential(ms) => ms.iter().flat_map(|m| m.parameters()).collect(),
            _ => vec![],
        }
    }
    /// `module.zero_grad()`
    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.t.zero_grad();
        }
    }
    /// `module.train()`
    pub fn train(&self) -> Module {
        self.clone()
    }
    /// `module.eval()`
    pub fn eval(&self) -> Module {
        self.clone()
    }
    /// `seq[i]`
    pub fn __getitem__(&self, i: i64) -> Result<Module, PyErr> {
        match &*self.0.borrow() {
            Layer::Sequential(ms) => {
                let n = ms.len() as i64;
                let j = if i < 0 { i + n } else { i };
                if j < 0 || j >= n {
                    return Err(PyErr::new(INDEX_ERROR, format!("index {} is out of range", i)));
                }
                Ok(ms[j as usize].clone())
            }
            _ => Err(PyErr::new(&["TypeError", "Exception", "BaseException"], format!("'{}' object is not subscriptable", self.name()))),
        }
    }
    /// `len(seq)`
    pub fn __len__(&self) -> i64 {
        match &*self.0.borrow() {
            Layer::Sequential(ms) => ms.len() as i64,
            _ => 0,
        }
    }
    /// `linear.weight`
    pub fn weight(&self) -> Result<Tensor, PyErr> {
        match &*self.0.borrow() {
            Layer::Linear { w, .. } => Ok(w.clone()),
            _ => Err(PyErr::new(ATTRIBUTE_ERROR, format!("'{}' object has no attribute 'weight'", self.name()))),
        }
    }
    /// `linear.bias`
    pub fn bias(&self) -> Result<Option<Tensor>, PyErr> {
        match &*self.0.borrow() {
            Layer::Linear { b, .. } => Ok(b.clone()),
            _ => Err(PyErr::new(ATTRIBUTE_ERROR, format!("'{}' object has no attribute 'bias'", self.name()))),
        }
    }
}

/// `nn.Linear(in_features, out_features, bias=True)`: PyTorch's
/// `kaiming_uniform_(a=√5)` weights and uniform bias from its generator.
pub fn linear(i: i64, o: i64, bias: bool) -> Result<Module, PyErr> {
    if i < 0 || o < 0 {
        return Err(rerr(format!("Trying to create tensor with negative dimension {}: [{}, {}]", i.min(o), o, i)));
    }
    let (i, o) = (i as usize, o as usize);
    let a = 5f64.sqrt();
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (i as f64).sqrt();
    let bound = 3f64.sqrt() * std;
    let w = param(uniform(i * o, -bound, bound), &[o, i]);
    let b = if bias {
        let bb = if i > 0 { 1.0 / (i as f64).sqrt() } else { 0.0 };
        Some(param(uniform(o, -bb, bb), &[o]))
    } else {
        None
    };
    Ok(Module::of(Layer::Linear { w, b, i, o }))
}
/// `nn.ReLU()`
pub fn relu() -> Module {
    Module::of(Layer::ReLU)
}
/// `nn.Sigmoid()`
pub fn sigmoid() -> Module {
    Module::of(Layer::Sigmoid)
}
/// `nn.Tanh()`
pub fn tanh() -> Module {
    Module::of(Layer::Tanh)
}
/// `nn.Softmax(dim)`
pub fn softmax(dim: i64) -> Module {
    Module::of(Layer::Softmax(dim))
}
/// `nn.Flatten()`
pub fn flatten() -> Module {
    Module::of(Layer::Flatten)
}
/// `nn.Identity()`
pub fn identity() -> Module {
    Module::of(Layer::Identity)
}
/// `nn.Sequential(*modules)`
pub fn sequential(ms: &[Module]) -> Module {
    Module::of(Layer::Sequential(ms.to_vec()))
}

// ------------------------------------------------------------ losses

#[derive(Clone, Copy, PartialEq, Debug)]
enum LossKind {
    Mse,
    L1,
    CrossEntropy,
}

/// `nn.MSELoss()`, `nn.L1Loss()`, `nn.CrossEntropyLoss()`
#[derive(Clone, PartialEq, Debug)]
pub struct Loss(LossKind);

impl Default for Loss {
    fn default() -> Loss {
        Loss(LossKind::Mse)
    }
}

fn check_same(a: &Tensor, b: &Tensor) -> Result<(), PyErr> {
    if a.dims() != b.dims() {
        return Err(rerr(format!(
            "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension {}",
            a.dims().last().unwrap_or(&1),
            b.dims().last().unwrap_or(&1),
            a.dims().len().max(1) - 1
        )));
    }
    Ok(())
}

/// `F.mse_loss(input, target)`
pub fn mse_loss(a: &Tensor, b: &Tensor) -> Result<Tensor, PyErr> {
    check_same(a, b)?;
    Ok(Tensor::op(F::mse_loss(&a.t, &b.t), DType::Float32, "MseLossBackward0"))
}
/// `F.l1_loss(input, target)`
pub fn l1_loss(a: &Tensor, b: &Tensor) -> Result<Tensor, PyErr> {
    check_same(a, b)?;
    Ok(Tensor::op((&a.t - &b.t).abs().mean_all(), DType::Float32, "MeanBackward0"))
}
/// `F.cross_entropy(logits, target)`: logits (N, C), int64 targets (N,)
pub fn cross_entropy(a: &Tensor, b: &Tensor) -> Result<Tensor, PyErr> {
    let (d, t) = (a.dims(), b.dims());
    if d.len() != 2 || t.len() != 1 {
        return Err(rerr("ruai: cross_entropy takes logits (N, C) and int64 targets (N,)"));
    }
    if b.dt != DType::Int64 {
        return Err(rerr(format!("expected scalar type Long but found {}", if b.dt == DType::Float32 { "Float" } else { "Bool" })));
    }
    if d[0] != t[0] {
        return Err(PyErr::new(&["ValueError", "Exception", "BaseException"], format!("Expected input batch_size ({}) to match target batch_size ({}).", d[0], t[0])));
    }
    let mut idx = vec![];
    for v in b.t.to_vec() {
        if v < 0.0 || v as usize >= d[1] {
            return Err(PyErr::new(INDEX_ERROR, format!("Target {} is out of bounds.", v as i64)));
        }
        idx.push(v as usize);
    }
    Ok(Tensor::op(F::cross_entropy(&a.t, &idx, 0.0), DType::Float32, "NllLossBackward0"))
}

impl Loss {
    /// `repr(loss)`
    pub fn __repr__(&self) -> String {
        match self.0 {
            LossKind::Mse => "MSELoss()",
            LossKind::L1 => "L1Loss()",
            LossKind::CrossEntropy => "CrossEntropyLoss()",
        }
        .into()
    }
    /// `str(loss)`
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
    /// `loss(input, target)`
    pub fn __call__(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, PyErr> {
        match self.0 {
            LossKind::Mse => mse_loss(a, b),
            LossKind::L1 => l1_loss(a, b),
            LossKind::CrossEntropy => cross_entropy(a, b),
        }
    }
}
/// `nn.MSELoss()`
pub fn mse() -> Loss {
    Loss(LossKind::Mse)
}
/// `nn.L1Loss()`
pub fn l1() -> Loss {
    Loss(LossKind::L1)
}
/// `nn.CrossEntropyLoss()`
pub fn cross_entropy_loss() -> Loss {
    Loss(LossKind::CrossEntropy)
}

// ------------------------------------------------------------ optimizers

enum Kind {
    Sgd { momentum: f64, bufs: Vec<Option<Vec<f32>>> },
    Adam { b1: f64, b2: f64, eps: f64, step: Vec<i64>, m: Vec<Vec<f32>>, v: Vec<Vec<f32>> },
}

struct OptState {
    params: Vec<Tensor>,
    lr: f64,
    wd: f64,
    kind: Kind,
}

/// `torch.optim.SGD` / `torch.optim.Adam` (PyTorch's single-tensor updates).
#[derive(Clone)]
pub struct Optimizer(Rc<RefCell<OptState>>);

impl Default for Optimizer {
    fn default() -> Optimizer {
        sgd(&[], 0.0, 0.0, 0.0)
    }
}

impl PartialEq for Optimizer {
    fn eq(&self, o: &Optimizer) -> bool {
        Rc::ptr_eq(&self.0, &o.0)
    }
}

impl std::fmt::Debug for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("Optimizer")
    }
}

/// `torch.optim.SGD(params, lr, momentum=0, weight_decay=0)`
pub fn sgd(params: &[Tensor], lr: f64, momentum: f64, weight_decay: f64) -> Optimizer {
    Optimizer(Rc::new(RefCell::new(OptState {
        params: params.to_vec(),
        lr,
        wd: weight_decay,
        kind: Kind::Sgd { momentum, bufs: vec![None; params.len()] },
    })))
}

/// `torch.optim.Adam(params, lr=0.001, weight_decay=0)`
pub fn adam(params: &[Tensor], lr: f64, weight_decay: f64) -> Optimizer {
    let n = params.len();
    Optimizer(Rc::new(RefCell::new(OptState {
        params: params.to_vec(),
        lr,
        wd: weight_decay,
        kind: Kind::Adam {
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            step: vec![0; n],
            m: params.iter().map(|p| vec![0.0; p.numel() as usize]).collect(),
            v: params.iter().map(|p| vec![0.0; p.numel() as usize]).collect(),
        },
    })))
}

impl Optimizer {
    /// `opt.zero_grad()`: the gradients become None
    pub fn zero_grad(&self) {
        for p in &self.0.borrow().params {
            p.t.zero_grad();
        }
    }
    /// `opt.step()`
    pub fn step(&self) {
        let mut st = self.0.borrow_mut();
        let (lr, wd, lr64) = (st.lr as f32, st.wd as f32, st.lr);
        let params = st.params.clone();
        for (k, p) in params.iter().enumerate() {
            let Some(g) = p.t.grad() else { continue };
            let mut x = p.t.to_vec();
            let mut d: Vec<f32> = g.to_vec();
            if wd != 0.0 {
                for (dj, xj) in d.iter_mut().zip(&x) {
                    *dj += wd * xj;
                }
            }
            match &mut st.kind {
                Kind::Sgd { momentum, bufs } => {
                    if *momentum != 0.0 {
                        let mo = *momentum as f32;
                        let buf = match bufs[k].take() {
                            None => d.clone(),
                            Some(mut b) => {
                                for (bj, dj) in b.iter_mut().zip(&d) {
                                    *bj = *bj * mo + dj;
                                }
                                b
                            }
                        };
                        d = buf.clone();
                        bufs[k] = Some(buf);
                    }
                    for (xj, dj) in x.iter_mut().zip(&d) {
                        *xj = dj.mul_add(-lr, *xj);
                    }
                }
                Kind::Adam { b1, b2, eps, step, m, v } => {
                    step[k] += 1;
                    let s = step[k] as i32;
                    let w = (1.0 - *b1) as f32;
                    let vb = (1.0 - *b2) as f32;
                    let b2f = *b2 as f32;
                    for j in 0..x.len() {
                        m[k][j] += w * (d[j] - m[k][j]);
                        v[k][j] = v[k][j] * b2f + vb * d[j] * d[j];
                    }
                    let bc1 = 1.0 - b1.powi(s);
                    let bc2 = 1.0 - b2.powi(s);
                    let step_size = (lr64 / bc1) as f32;
                    let bc2s = bc2.sqrt() as f32;
                    let e = *eps as f32;
                    for j in 0..x.len() {
                        let denom = v[k][j].sqrt() / bc2s + e;
                        x[j] += -step_size * (m[k][j] / denom);
                    }
                }
            }
            p.t.copy_(&T::from_vec(x, &p.dims(), Device::Cpu));
        }
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

py_text!(Module, Loss);
