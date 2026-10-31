//! `gymnasium` for py0 (`import gymnasium as gym`): the environments of
//! `ruai::world`, which match Gymnasium bit for bit, with Gymnasium's Python
//! API: `gym.make(id)`, `env.reset(seed=...)`, `env.step(action)`,
//! `env.action_space.sample()`. Observations are numpy arrays (float32 for
//! `Box` spaces, a 0-d int64 for `Discrete` ones); rewards are floats.

use super::numpy::{NdArray, Operand};
use crate::pyrt::err::{PyErr, VALUE_ERROR};
use crate::world::{self, SpaceKind, Value};
use indexmap::IndexMap;
use std::cell::RefCell;
use std::rc::Rc;

const NAME_NOT_FOUND: &[&str] = &["NameNotFound", "UnregisteredEnv", "Error", "Exception", "BaseException"];

type Shared = Rc<RefCell<Box<dyn world::Env>>>;

/// `gymnasium.Env` (made by `gym.make`, wrapped in its `TimeLimit`).
#[derive(Clone)]
pub struct Env {
    env: Shared,
    id: String,
    class: &'static str,
    ansi: bool,
}

impl Default for Env {
    fn default() -> Env {
        make("CartPole-v1", &None).unwrap()
    }
}

impl PartialEq for Env {
    fn eq(&self, o: &Env) -> bool {
        Rc::ptr_eq(&self.env, &o.env)
    }
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.__str__())
    }
}

fn to_array(v: &Value, shape: &[usize]) -> NdArray {
    match v {
        Value::Int(i) => NdArray::from_scalar(super::numpy::Scalar::I(*i)),
        Value::Floats(x) => super::numpy::array(x.iter().map(|&f| f as f64).collect::<Vec<f64>>(), &Some("float32".into()))
            .unwrap()
            .reshape(shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())
            .unwrap(),
    }
}

fn info(i: &world::Info) -> IndexMap<String, f64> {
    i.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// `gym.make(id, render_mode=None)`
pub fn make(id: &str, render_mode: &Option<String>) -> Result<Env, PyErr> {
    let env = world::make(id).map_err(|_| {
        let name = id.split('-').next().unwrap_or(id);
        PyErr::new(NAME_NOT_FOUND, format!("Environment `{}` doesn't exist.", name))
    })?;
    let class = match id {
        "CartPole-v1" => "CartPoleEnv",
        "MountainCar-v0" => "MountainCarEnv",
        "Pendulum-v1" => "PendulumEnv",
        "Acrobot-v1" => "AcrobotEnv",
        _ => "FrozenLakeEnv",
    };
    let ansi = match render_mode.as_deref() {
        None => false,
        Some("ansi") if class == "FrozenLakeEnv" => true,
        Some(m) => return Err(PyErr::new(VALUE_ERROR, format!("ruai: render_mode={:?} is not supported (FrozenLake has \"ansi\")", m))),
    };
    Ok(Env { env: Rc::new(RefCell::new(env)), id: id.to_string(), class, ansi })
}

/// The action given to `step`: an int, or an array (for Box actions).
pub trait Action {
    fn value(&self, space: &world::Space) -> Result<Value, PyErr>;
}

impl Action for i64 {
    fn value(&self, _: &world::Space) -> Result<Value, PyErr> {
        Ok(Value::Int(*self))
    }
}

impl Action for &NdArray {
    fn value(&self, space: &world::Space) -> Result<Value, PyErr> {
        match space.kind {
            SpaceKind::Discrete { .. } => Ok(Value::Int(self.__int__()?)),
            SpaceKind::Box { .. } => Ok(Value::Floats(self.to_f64().into_iter().map(|x| x as f32).collect())),
        }
    }
}

impl Env {
    /// `str(env)`: Gymnasium's wrapper chain.
    pub fn __str__(&self) -> String {
        format!("<TimeLimit<OrderEnforcing<PassiveEnvChecker<{}<{}>>>>>", self.class, self.id)
    }
    /// `repr(env)`
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
    /// `env.reset(seed=None)`: (observation, info)
    pub fn reset(&self, seed: Option<i64>) -> Result<(NdArray, IndexMap<String, f64>), PyErr> {
        if seed.is_some_and(|s| s < 0) {
            return Err(PyErr::new(VALUE_ERROR, "Seed must be a non-negative integer or omitted, not a negative value."));
        }
        let mut e = self.env.borrow_mut();
        let (obs, i) = e.reset(seed.map(|s| s as u64));
        let shape = e.observation_space().kind.clone();
        Ok((to_array(&obs, &box_shape(&shape)), info(&i)))
    }
    /// `env.step(action)`: (observation, reward, terminated, truncated, info)
    pub fn step(&self, action: impl Action) -> Result<(NdArray, f64, bool, bool, IndexMap<String, f64>), PyErr> {
        let mut e = self.env.borrow_mut();
        let a = action.value(e.action_space())?;
        if !e.action_space().contains(&a) {
            return Err(PyErr::new(&["AssertionError", "Exception", "BaseException"], format!("{:?} ({}) invalid", a, "action")));
        }
        let s = e.step(&a);
        let shape = e.observation_space().kind.clone();
        Ok((to_array(&s.obs, &box_shape(&shape)), s.reward, s.terminated, s.truncated, info(&s.info)))
    }
    /// `env.render()` (FrozenLake with `render_mode="ansi"`)
    pub fn render(&self) -> Result<String, PyErr> {
        if !self.ansi {
            return Err(PyErr::new(VALUE_ERROR, "ruai: render() needs render_mode=\"ansi\" (FrozenLake)"));
        }
        Ok(self.env.borrow().render())
    }
    /// `env.close()`
    pub fn close(&self) {}
    /// `env.action_space`
    pub fn action_space(&self) -> Space {
        Space { env: self.env.clone(), action: true }
    }
    /// `env.observation_space`
    pub fn observation_space(&self) -> Space {
        Space { env: self.env.clone(), action: false }
    }
}

fn box_shape(k: &SpaceKind) -> Vec<usize> {
    match k {
        SpaceKind::Box { shape, .. } => shape.clone(),
        SpaceKind::Discrete { .. } => vec![],
    }
}

/// `env.action_space` / `env.observation_space` (a `Discrete` or a `Box`).
#[derive(Clone)]
pub struct Space {
    env: Shared,
    action: bool,
}

impl Default for Space {
    fn default() -> Space {
        Env::default().action_space()
    }
}

impl PartialEq for Space {
    fn eq(&self, o: &Space) -> bool {
        Rc::ptr_eq(&self.env, &o.env) && self.action == o.action
    }
}

impl std::fmt::Debug for Space {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.__str__())
    }
}

fn f32_array(v: &[f32]) -> NdArray {
    super::numpy::array(v.iter().map(|&f| f as f64).collect::<Vec<f64>>(), &Some("float32".into())).unwrap()
}

impl Space {
    fn kind(&self) -> SpaceKind {
        let e = self.env.borrow();
        if self.action {
            e.action_space().kind.clone()
        } else {
            e.observation_space().kind.clone()
        }
    }
    /// `str(space)`: `Discrete(2)`, `Box(-2.0, 2.0, (1,), float32)`
    pub fn __str__(&self) -> String {
        match self.kind() {
            SpaceKind::Discrete { n, start: 0 } => format!("Discrete({})", n),
            SpaceKind::Discrete { n, start } => format!("Discrete({}, start={})", n, start),
            SpaceKind::Box { low, high, shape } => {
                let same = |v: &[f32]| v.iter().all(|x| *x == v[0]);
                let side = |v: &[f32]| {
                    if same(v) {
                        crate::pyrt::float32_repr(v[0])
                    } else {
                        f32_array(v).__str__()
                    }
                };
                let sh = super::numpy::Shape(shape.iter().map(|&d| d as i64).collect());
                format!("Box({}, {}, {}, float32)", side(&low), side(&high), sh.__str__())
            }
        }
    }
    /// `repr(space)`
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
    /// `space.n` (Discrete)
    pub fn n(&self) -> Result<NdArray, PyErr> {
        match self.kind() {
            SpaceKind::Discrete { n, .. } => Ok(NdArray::from_scalar(super::numpy::Scalar::I(n))),
            SpaceKind::Box { .. } => Err(PyErr::new(&["AttributeError", "Exception", "BaseException"], "'Box' object has no attribute 'n'")),
        }
    }
    /// `space.shape`
    pub fn shape(&self) -> super::numpy::Shape {
        super::numpy::Shape(box_shape(&self.kind()).iter().map(|&d| d as i64).collect())
    }
    fn bound(&self, high: bool) -> Result<NdArray, PyErr> {
        match self.kind() {
            SpaceKind::Box { low, high: h, shape } => {
                Ok(f32_array(if high { &h } else { &low }).reshape(shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())?)
            }
            SpaceKind::Discrete { .. } => Err(PyErr::new(&["AttributeError", "Exception", "BaseException"], "'Discrete' object has no attribute 'low'")),
        }
    }
    /// `space.low` (Box)
    pub fn low(&self) -> Result<NdArray, PyErr> {
        self.bound(false)
    }
    /// `space.high` (Box)
    pub fn high(&self) -> Result<NdArray, PyErr> {
        self.bound(true)
    }
    fn with<R>(&self, f: impl FnOnce(&mut world::Space) -> R) -> R {
        let mut e = self.env.borrow_mut();
        if self.action {
            f(e.action_space_mut())
        } else {
            f(e.observation_space_mut())
        }
    }
    /// `space.seed(seed)`
    pub fn seed(&self, seed: i64) -> i64 {
        self.with(|s| s.seed(seed as u64));
        seed
    }
    /// `space.sample()`: an int64 scalar (Discrete) or a float32 array (Box)
    pub fn sample(&self) -> Result<NdArray, PyErr> {
        let shape = box_shape(&self.kind());
        if let SpaceKind::Box { low, high, .. } = self.kind() {
            if low.iter().chain(&high).any(|x| !x.is_finite()) {
                return Err(PyErr::new(VALUE_ERROR, "ruai: sampling an unbounded Box is not supported"));
            }
        }
        let v = self.with(|s| s.sample());
        Ok(to_array(&v, &shape))
    }
    /// `space.contains(x)`
    pub fn contains(&self, x: impl Operand + Action) -> Result<bool, PyErr> {
        let e = self.env.borrow();
        let sp = if self.action { e.action_space() } else { e.observation_space() };
        Ok(sp.contains(&x.value(sp)?))
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

py_text!(Env, Space);
