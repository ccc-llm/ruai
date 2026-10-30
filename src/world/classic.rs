//! Gymnasium's classic-control environments. Each step repeats Gymnasium's
//! arithmetic in the same order and precision (including where NumPy works
//! in float32), so results match bit for bit.

use super::{Env, EnvRng, Info, Space, Step, Value};
use std::f64::consts::PI;

/// NumPy's scalar `x ** 2`: the C library's `pow` (not always the correctly
/// rounded `x * x` that NumPy uses for arrays). `black_box` keeps LLVM from
/// rewriting `pow(x, 2.0)` into a multiplication.
fn sq(x: f64) -> f64 {
    x.powf(std::hint::black_box(2.0))
}

/// [`sq`] for a float32 scalar (`powf`).
fn sq32(x: f32) -> f32 {
    x.powf(std::hint::black_box(2.0))
}

fn f32s(xs: &[f64]) -> Value {
    Value::Floats(xs.iter().map(|&x| x as f32).collect())
}

/// `CartPole-v1`: keep a pole upright by pushing the cart left (0) or right (1).
pub struct CartPole {
    state: Option<[f64; 4]>,
    steps_beyond_terminated: Option<usize>,
    rng: EnvRng,
    obs_space: Space,
    act_space: Space,
}

const CP_GRAVITY: f64 = 9.8;
const CP_MASSCART: f64 = 1.0;
const CP_MASSPOLE: f64 = 0.1;
const CP_LENGTH: f64 = 0.5;
const CP_FORCE: f64 = 10.0;
const CP_TAU: f64 = 0.02;
const CP_X_THRESHOLD: f64 = 2.4;

impl CartPole {
    /// A new environment (call [`Env::reset`] first).
    pub fn new() -> Self {
        let theta = 12.0 * 2.0 * PI / 360.0;
        let high = vec![(CP_X_THRESHOLD * 2.0) as f32, f32::INFINITY, (theta * 2.0) as f32, f32::INFINITY];
        CartPole {
            state: None,
            steps_beyond_terminated: None,
            rng: EnvRng::default(),
            obs_space: Space::boxed(high.iter().map(|h| -h).collect(), high, vec![4]),
            act_space: Space::discrete(2),
        }
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Env for CartPole {
    fn id(&self) -> &str {
        "CartPole-v1"
    }
    fn observation_space(&self) -> &Space {
        &self.obs_space
    }
    fn action_space(&self) -> &Space {
        &self.act_space
    }
    fn action_space_mut(&mut self) -> &mut Space {
        &mut self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.rng.reseed(seed);
        let g = self.rng.get();
        let s = [g.uniform(-0.05, 0.05), g.uniform(-0.05, 0.05), g.uniform(-0.05, 0.05), g.uniform(-0.05, 0.05)];
        self.state = Some(s);
        self.steps_beyond_terminated = None;
        (f32s(&s), Info::new())
    }

    fn step(&mut self, action: &Value) -> Step {
        assert!(self.act_space.contains(action), "{action:?} invalid");
        let [x, x_dot, theta, theta_dot] = self.state.expect("call reset before step");
        let total_mass = CP_MASSPOLE + CP_MASSCART;
        let polemass_length = CP_MASSPOLE * CP_LENGTH;
        let force = if action.int() == 1 { CP_FORCE } else { -CP_FORCE };
        let (costheta, sintheta) = (theta.cos(), theta.sin());
        let temp = (force + polemass_length * (theta_dot * theta_dot) * sintheta) / total_mass;
        let thetaacc = (CP_GRAVITY * sintheta - costheta * temp)
            / (CP_LENGTH * (4.0 / 3.0 - CP_MASSPOLE * (costheta * costheta) / total_mass));
        let xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        // Euler integration
        let s = [x + CP_TAU * x_dot, x_dot + CP_TAU * xacc, theta + CP_TAU * theta_dot, theta_dot + CP_TAU * thetaacc];
        self.state = Some(s);
        let theta_threshold = 12.0 * 2.0 * PI / 360.0;
        let terminated = s[0] < -CP_X_THRESHOLD || s[0] > CP_X_THRESHOLD || s[2] < -theta_threshold || s[2] > theta_threshold;
        let reward = if !terminated {
            1.0
        } else if self.steps_beyond_terminated.is_none() {
            self.steps_beyond_terminated = Some(0);
            1.0
        } else {
            *self.steps_beyond_terminated.as_mut().unwrap() += 1;
            0.0
        };
        Step { obs: f32s(&s), reward, terminated, truncated: false, info: Info::new() }
    }

    fn render(&self) -> String {
        match self.state {
            Some([x, xd, th, thd]) => format!("cart x={x:+.3} v={xd:+.3}  pole θ={th:+.3} ω={thd:+.3}\n"),
            None => "(not reset)\n".into(),
        }
    }
}

/// `MountainCar-v0`: accelerate left (0), not at all (1) or right (2) to reach the flag.
pub struct MountainCar {
    state: Option<(f64, f64)>,
    rng: EnvRng,
    obs_space: Space,
    act_space: Space,
}

const MC_MIN_POS: f64 = -1.2;
const MC_MAX_POS: f64 = 0.6;
const MC_MAX_SPEED: f64 = 0.07;
const MC_GOAL: f64 = 0.5;

impl MountainCar {
    /// A new environment (call [`Env::reset`] first).
    pub fn new() -> Self {
        MountainCar {
            state: None,
            rng: EnvRng::default(),
            obs_space: Space::boxed(
                vec![MC_MIN_POS as f32, -MC_MAX_SPEED as f32],
                vec![MC_MAX_POS as f32, MC_MAX_SPEED as f32],
                vec![2],
            ),
            act_space: Space::discrete(3),
        }
    }
}

impl Default for MountainCar {
    fn default() -> Self {
        Self::new()
    }
}

impl Env for MountainCar {
    fn id(&self) -> &str {
        "MountainCar-v0"
    }
    fn observation_space(&self) -> &Space {
        &self.obs_space
    }
    fn action_space(&self) -> &Space {
        &self.act_space
    }
    fn action_space_mut(&mut self) -> &mut Space {
        &mut self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.rng.reseed(seed);
        let s = (self.rng.get().uniform(-0.6, -0.4), 0.0);
        self.state = Some(s);
        (f32s(&[s.0, s.1]), Info::new())
    }

    fn step(&mut self, action: &Value) -> Step {
        assert!(self.act_space.contains(action), "{action:?} invalid");
        let (mut position, mut velocity) = self.state.expect("call reset before step");
        velocity += (action.int() - 1) as f64 * 0.001 + (3.0 * position).cos() * (-0.0025);
        velocity = velocity.clamp(-MC_MAX_SPEED, MC_MAX_SPEED);
        position += velocity;
        position = position.clamp(MC_MIN_POS, MC_MAX_POS);
        if position == MC_MIN_POS && velocity < 0.0 {
            velocity = 0.0;
        }
        let terminated = position >= MC_GOAL && velocity >= 0.0;
        self.state = Some((position, velocity));
        Step { obs: f32s(&[position, velocity]), reward: -1.0, terminated, truncated: false, info: Info::new() }
    }

    fn render(&self) -> String {
        match self.state {
            Some((p, v)) => {
                let col = ((p - MC_MIN_POS) / (MC_MAX_POS - MC_MIN_POS) * 40.0).round() as usize;
                let mut track: Vec<char> = "_".repeat(41).chars().collect();
                track[((MC_GOAL - MC_MIN_POS) / (MC_MAX_POS - MC_MIN_POS) * 40.0).round() as usize] = '|';
                track[col.min(40)] = 'o';
                format!("{}  x={p:+.3} v={v:+.4}\n", track.into_iter().collect::<String>())
            }
            None => "(not reset)\n".into(),
        }
    }
}

/// `Pendulum-v1`: swing up and balance with a torque in `[-2, 2]` (a `Box` of one float).
pub struct Pendulum {
    g: f64,
    state: Option<[f64; 2]>,
    rng: EnvRng,
    obs_space: Space,
    act_space: Space,
}

fn angle_normalize(x: f64) -> f64 {
    // NumPy's float remainder: fmod, shifted into the divisor's sign
    let b = 2.0 * PI;
    let mut m = (x + PI) % b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        m += b;
    }
    m - PI
}

impl Pendulum {
    /// A new environment with gravity `g` (Gymnasium's default is 10).
    pub fn new(g: f64) -> Self {
        Pendulum {
            g,
            state: None,
            rng: EnvRng::default(),
            obs_space: Space::boxed(vec![-1.0, -1.0, -8.0], vec![1.0, 1.0, 8.0], vec![3]),
            act_space: Space::boxed(vec![-2.0], vec![2.0], vec![1]),
        }
    }

    /// The internal state `[theta, theta_dot]` (Gymnasium's `unwrapped.state`).
    pub fn state(&self) -> Option<[f64; 2]> {
        self.state
    }

    fn obs(&self) -> Value {
        let [th, thdot] = self.state.unwrap();
        f32s(&[th.cos(), th.sin(), thdot])
    }
}

impl Env for Pendulum {
    fn id(&self) -> &str {
        "Pendulum-v1"
    }
    fn observation_space(&self) -> &Space {
        &self.obs_space
    }
    fn action_space(&self) -> &Space {
        &self.act_space
    }
    fn action_space_mut(&mut self) -> &mut Space {
        &mut self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.rng.reseed(seed);
        let g = self.rng.get();
        self.state = Some([g.uniform(-PI, PI), g.uniform(-1.0, 1.0)]);
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: &Value) -> Step {
        let [th, thdot] = self.state.expect("call reset before step");
        // the action is float32, so NumPy evaluates the terms with `u` in float32
        let u: f32 = action.floats()[0].clamp(-2.0, 2.0);
        let costs = sq(angle_normalize(th)) + 0.1 * sq(thdot) + (0.001f32 * sq32(u)) as f64;
        let (dt, l, m) = (0.05, 1.0, 1.0);
        let newthdot = thdot + (3.0 * self.g / (2.0 * l) * th.sin() + (3.0f32 / (m * l * l) as f32 * u) as f64) * dt;
        let newthdot = newthdot.clamp(-8.0, 8.0);
        let newth = th + newthdot * dt;
        self.state = Some([newth, newthdot]);
        Step { obs: self.obs(), reward: -costs, terminated: false, truncated: false, info: Info::new() }
    }

    fn render(&self) -> String {
        match self.state {
            Some([th, thd]) => format!("pendulum θ={:+.3} ω={thd:+.3}\n", angle_normalize(th)),
            None => "(not reset)\n".into(),
        }
    }
}

/// `Acrobot-v1`: swing the end of a two-link arm above the line with torque -1, 0 or +1.
pub struct Acrobot {
    state: Option<[f64; 4]>,
    rng: EnvRng,
    obs_space: Space,
    act_space: Space,
}

const AC_DT: f64 = 0.2;
const AC_MAX_VEL_1: f64 = 4.0 * PI;
const AC_MAX_VEL_2: f64 = 9.0 * PI;

fn wrap(mut x: f64, m: f64, mm: f64) -> f64 {
    let diff = mm - m;
    while x > mm {
        x -= diff;
    }
    while x < m {
        x += diff;
    }
    x
}

/// Derivatives of `[theta1, theta2, dtheta1, dtheta2, torque]` ("book" dynamics).
fn acrobot_dsdt(s: &[f64; 5]) -> [f64; 5] {
    let (m1, m2, l1, lc1, lc2, i1, i2, g) = (1.0f64, 1.0f64, 1.0f64, 0.5f64, 0.5f64, 1.0f64, 1.0f64, 9.8f64);
    let a = s[4];
    let (theta1, theta2, dtheta1, dtheta2) = (s[0], s[1], s[2], s[3]);
    // the constants' squares are exact; the state's go through pow like NumPy's scalars
    let d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * theta2.cos()) + i1 + i2;
    let d2 = m2 * (lc2 * lc2 + l1 * lc2 * theta2.cos()) + i2;
    let phi2 = m2 * lc2 * g * (theta1 + theta2 - PI / 2.0).cos();
    let phi1 = -m2 * l1 * lc2 * sq(dtheta2) * theta2.sin() - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * theta2.sin()
        + (m1 * lc1 + m2 * l1) * g * (theta1 - PI / 2.0).cos()
        + phi2;
    let ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * sq(dtheta1) * theta2.sin() - phi2) / (m2 * lc2 * lc2 + i2 - sq(d2) / d1);
    let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
    [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]
}

impl Acrobot {
    /// A new environment (call [`Env::reset`] first).
    pub fn new() -> Self {
        let high = vec![1.0, 1.0, 1.0, 1.0, AC_MAX_VEL_1 as f32, AC_MAX_VEL_2 as f32];
        Acrobot {
            state: None,
            rng: EnvRng::default(),
            obs_space: Space::boxed(high.iter().map(|h| -h).collect(), high, vec![6]),
            act_space: Space::discrete(3),
        }
    }

    fn obs(&self) -> Value {
        let s = self.state.unwrap();
        f32s(&[s[0].cos(), s[0].sin(), s[1].cos(), s[1].sin(), s[2], s[3]])
    }
}

impl Default for Acrobot {
    fn default() -> Self {
        Self::new()
    }
}

impl Env for Acrobot {
    fn id(&self) -> &str {
        "Acrobot-v1"
    }
    fn observation_space(&self) -> &Space {
        &self.obs_space
    }
    fn action_space(&self) -> &Space {
        &self.act_space
    }
    fn action_space_mut(&mut self) -> &mut Space {
        &mut self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.rng.reseed(seed);
        let g = self.rng.get();
        // Gymnasium stores the initial state as float32
        let s = [(); 4].map(|_| g.uniform(-0.1, 0.1) as f32 as f64);
        self.state = Some(s);
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: &Value) -> Step {
        assert!(self.act_space.contains(action), "{action:?} invalid");
        let s = self.state.expect("call reset before step");
        let torque = [-1.0, 0.0, 1.0][action.int() as usize];
        // one RK4 step over [0, dt]
        let y0 = [s[0], s[1], s[2], s[3], torque];
        let (dt, dt2) = (AC_DT, AC_DT / 2.0);
        let add = |y: &[f64; 5], h: f64, k: &[f64; 5]| -> [f64; 5] { std::array::from_fn(|i| y[i] + h * k[i]) };
        let k1 = acrobot_dsdt(&y0);
        let k2 = acrobot_dsdt(&add(&y0, dt2, &k1));
        let k3 = acrobot_dsdt(&add(&y0, dt2, &k2));
        let k4 = acrobot_dsdt(&add(&y0, dt, &k3));
        let ns: [f64; 5] = std::array::from_fn(|i| y0[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
        let ns = [
            wrap(ns[0], -PI, PI),
            wrap(ns[1], -PI, PI),
            ns[2].clamp(-AC_MAX_VEL_1, AC_MAX_VEL_1),
            ns[3].clamp(-AC_MAX_VEL_2, AC_MAX_VEL_2),
        ];
        self.state = Some(ns);
        let terminated = -ns[0].cos() - (ns[1] + ns[0]).cos() > 1.0;
        Step { obs: self.obs(), reward: if terminated { 0.0 } else { -1.0 }, terminated, truncated: false, info: Info::new() }
    }

    fn render(&self) -> String {
        match self.state {
            Some(s) => format!("acrobot θ1={:+.3} θ2={:+.3} ω1={:+.3} ω2={:+.3}  height={:+.3}\n", s[0], s[1], s[2], s[3], -s[0].cos() - (s[1] + s[0]).cos()),
            None => "(not reset)\n".into(),
        }
    }
}
