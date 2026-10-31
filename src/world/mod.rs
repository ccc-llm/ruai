//! Environments for reinforcement learning with the Gymnasium API (feature
//! `world`): [`make`] an environment by id, [`Env::reset`] it with a seed,
//! [`Env::step`] it with actions.
//!
//! The built-in environments reproduce Gymnasium 1.x exactly: same dynamics,
//! same floating-point operations, and the same random numbers for the same
//! seed (through [`crate::rng`], NumPy's `PCG64`), so a seeded episode gives
//! the observations, rewards and flags that `gymnasium.make(id)` gives.
//!
//! ```
//! use ruai::world::{self, Value};
//! let mut env = world::make("CartPole-v1").unwrap();
//! let (obs, _) = env.reset(Some(42));
//! assert_eq!(obs.floats().len(), 4);
//! let s = env.step(&Value::Int(1));
//! assert_eq!(s.reward, 1.0);
//! ```

mod classic;
mod frozen_lake;
mod wrappers;

pub use classic::{Acrobot, CartPole, MountainCar, Pendulum};
pub use frozen_lake::FrozenLake;
pub use wrappers::{RecordEpisodeStatistics, SyncVectorEnv, TimeLimit, VectorStep};

use crate::rng::Generator;
use std::collections::BTreeMap;

/// An observation or action: an integer (`Discrete` spaces) or a float32
/// vector (`Box` spaces, flattened row-major).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// An element of a `Discrete` space.
    Int(i64),
    /// An element of a `Box` space.
    Floats(Vec<f32>),
}

impl Value {
    /// The integer of a `Discrete` value (panics on floats).
    pub fn int(&self) -> i64 {
        match self {
            Value::Int(i) => *i,
            Value::Floats(_) => panic!("expected a discrete value, got {self:?}"),
        }
    }

    /// The floats of a `Box` value (panics on integers).
    pub fn floats(&self) -> &[f32] {
        match self {
            Value::Floats(v) => v,
            Value::Int(_) => panic!("expected a Box value, got {self:?}"),
        }
    }

    /// As a float vector: a `Box` value as is, an integer as `[i]`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Value::Int(i) => vec![*i as f32],
            Value::Floats(v) => v.clone(),
        }
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

impl From<usize> for Value {
    fn from(i: usize) -> Self {
        Value::Int(i as i64)
    }
}

impl From<Vec<f32>> for Value {
    fn from(v: Vec<f32>) -> Self {
        Value::Floats(v)
    }
}

/// Extra information from `reset` / `step` (Gymnasium's `info` dict, numbers only).
pub type Info = BTreeMap<String, f64>;

/// The result of [`Env::step`].
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// The next observation.
    pub obs: Value,
    /// The reward for this transition.
    pub reward: f64,
    /// The episode ended in a terminal state of the task.
    pub terminated: bool,
    /// The episode was cut off (e.g. by a time limit).
    pub truncated: bool,
    /// Extra information.
    pub info: Info,
}

/// The shape of a space.
#[derive(Clone, Debug, PartialEq)]
pub enum SpaceKind {
    /// Integers `start .. start + n`.
    Discrete {
        /// Number of elements.
        n: i64,
        /// Smallest element.
        start: i64,
    },
    /// float32 vectors with per-element bounds (infinite where unbounded).
    Box {
        /// Lower bounds.
        low: Vec<f32>,
        /// Upper bounds.
        high: Vec<f32>,
        /// Shape (the vector is its row-major flattening).
        shape: Vec<usize>,
    },
}

/// An observation or action space with its own random generator, like
/// `gymnasium.spaces.Space` (`seed`, `sample`, `contains`).
#[derive(Clone, Debug)]
pub struct Space {
    /// What the space contains.
    pub kind: SpaceKind,
    rng: Option<Generator>,
}

impl Space {
    /// `Discrete(n)`.
    pub fn discrete(n: i64) -> Self {
        Space { kind: SpaceKind::Discrete { n, start: 0 }, rng: None }
    }

    /// `Box(low, high, shape, float32)`.
    pub fn boxed(low: Vec<f32>, high: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(low.len(), high.len());
        assert_eq!(low.len(), shape.iter().product::<usize>());
        Space { kind: SpaceKind::Box { low, high, shape }, rng: None }
    }

    /// Seed the space's generator (`space.seed(seed)`).
    pub fn seed(&mut self, seed: u64) {
        self.rng = Some(Generator::new(seed));
    }

    fn rng(&mut self) -> &mut Generator {
        self.rng.get_or_insert_with(Generator::from_entropy)
    }

    /// A random element, drawn like Gymnasium's `sample()`.
    pub fn sample(&mut self) -> Value {
        match self.kind.clone() {
            SpaceKind::Discrete { n, start } => Value::Int(start + self.rng().integers(0, n)),
            SpaceKind::Box { low, high, .. } => {
                assert!(
                    low.iter().chain(&high).all(|x| x.is_finite()),
                    "sampling unbounded Box dimensions is not supported"
                );
                let v = low
                    .iter()
                    .zip(&high)
                    .map(|(&l, &h)| (self.rng().uniform(l as f64, h as f64) as f32).clamp(l, h))
                    .collect();
                Value::Floats(v)
            }
        }
    }

    /// Whether `v` is an element of the space.
    pub fn contains(&self, v: &Value) -> bool {
        match (&self.kind, v) {
            (SpaceKind::Discrete { n, start }, Value::Int(i)) => (*start..start + n).contains(i),
            (SpaceKind::Box { low, high, .. }, Value::Floats(x)) => {
                x.len() == low.len() && x.iter().zip(low.iter().zip(high)).all(|(x, (l, h))| l <= x && x <= h)
            }
            _ => false,
        }
    }

    /// Number of actions of a `Discrete` space, or the length of a `Box` vector.
    pub fn size(&self) -> usize {
        match &self.kind {
            SpaceKind::Discrete { n, .. } => *n as usize,
            SpaceKind::Box { low, .. } => low.len(),
        }
    }
}

impl std::fmt::Display for Space {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            SpaceKind::Discrete { n, start: 0 } => write!(f, "Discrete({n})"),
            SpaceKind::Discrete { n, start } => write!(f, "Discrete({n}, start={start})"),
            SpaceKind::Box { shape, .. } => write!(f, "Box{shape:?}"),
        }
    }
}

/// An environment's random generator: `reset(Some(seed))` re-seeds it,
/// otherwise it continues (seeded from entropy on first use).
#[derive(Clone, Debug, Default)]
pub struct EnvRng(Option<Generator>);

impl EnvRng {
    /// Gymnasium's `super().reset(seed=seed)`.
    pub fn reseed(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.0 = Some(Generator::new(s));
        }
    }

    /// The generator (`self.np_random`).
    pub fn get(&mut self) -> &mut Generator {
        self.0.get_or_insert_with(Generator::from_entropy)
    }
}

/// A reinforcement-learning environment (`gymnasium.Env`).
pub trait Env: Send {
    /// The registered id, e.g. `"CartPole-v1"`.
    fn id(&self) -> &str;
    /// The space of observations.
    fn observation_space(&self) -> &Space;
    /// The space of actions.
    fn action_space(&self) -> &Space;
    /// The space of actions, mutable (to seed or sample it).
    fn action_space_mut(&mut self) -> &mut Space;
    /// The observation space, to seed or sample it.
    fn observation_space_mut(&mut self) -> &mut Space;
    /// Start an episode; `Some(seed)` re-seeds the environment's generator.
    fn reset(&mut self, seed: Option<u64>) -> (Value, Info);
    /// Apply an action.
    fn step(&mut self, action: &Value) -> Step;
    /// A text picture of the current state (Gymnasium's `"ansi"` render
    /// mode where it has one).
    fn render(&self) -> String;
}

/// A registered environment: id, time limit and description.
pub struct Registration {
    /// The id for [`make`].
    pub id: &'static str,
    /// Steps before the episode is truncated.
    pub max_episode_steps: usize,
    /// One line about the task.
    pub description: &'static str,
}

/// The built-in environments.
pub const REGISTRY: &[Registration] = &[
    Registration { id: "CartPole-v1", max_episode_steps: 500, description: "balance a pole on a cart by pushing it left or right" },
    Registration { id: "MountainCar-v0", max_episode_steps: 200, description: "drive an underpowered car up a hill by rocking it" },
    Registration { id: "Pendulum-v1", max_episode_steps: 200, description: "swing a pendulum up and keep it upright with a continuous torque" },
    Registration { id: "Acrobot-v1", max_episode_steps: 500, description: "swing a two-link arm above a line by torquing its joint" },
    Registration { id: "FrozenLake-v1", max_episode_steps: 100, description: "cross a slippery 4x4 frozen lake without falling in a hole" },
    Registration { id: "FrozenLake8x8-v1", max_episode_steps: 200, description: "cross a slippery 8x8 frozen lake without falling in a hole" },
];

/// Create a registered environment wrapped in its [`TimeLimit`], like
/// `gymnasium.make(id)`.
pub fn make(id: &str) -> Result<Box<dyn Env>, String> {
    let reg = REGISTRY.iter().find(|r| r.id == id).ok_or_else(|| {
        let ids: Vec<&str> = REGISTRY.iter().map(|r| r.id).collect();
        format!("unknown environment '{id}' (available: {})", ids.join(", "))
    })?;
    let env: Box<dyn Env> = match id {
        "CartPole-v1" => Box::new(CartPole::new()),
        "MountainCar-v0" => Box::new(MountainCar::new()),
        "Pendulum-v1" => Box::new(Pendulum::new(10.0)),
        "Acrobot-v1" => Box::new(Acrobot::new()),
        "FrozenLake-v1" => Box::new(FrozenLake::new("4x4", true)),
        "FrozenLake8x8-v1" => Box::new(FrozenLake::new("8x8", true)),
        _ => unreachable!(),
    };
    Ok(Box::new(TimeLimit::new(env, reg.max_episode_steps)))
}
