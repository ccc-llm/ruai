//! Gymnasium's wrappers: time limits, episode statistics and a synchronous
//! vector environment.

use super::{Env, Info, Space, Step, Value};

/// Truncate episodes after `max_episode_steps` steps (`gymnasium.wrappers.TimeLimit`).
pub struct TimeLimit {
    env: Box<dyn Env>,
    max_episode_steps: usize,
    elapsed: usize,
}

impl TimeLimit {
    /// Wrap `env`.
    pub fn new(env: Box<dyn Env>, max_episode_steps: usize) -> Self {
        TimeLimit { env, max_episode_steps, elapsed: 0 }
    }

    /// Steps since the last reset.
    pub fn elapsed_steps(&self) -> usize {
        self.elapsed
    }
}

impl Env for TimeLimit {
    fn id(&self) -> &str {
        self.env.id()
    }
    fn observation_space(&self) -> &Space {
        self.env.observation_space()
    }
    fn action_space(&self) -> &Space {
        self.env.action_space()
    }
    fn action_space_mut(&mut self) -> &mut Space {
        self.env.action_space_mut()
    }
    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.elapsed = 0;
        self.env.reset(seed)
    }
    fn step(&mut self, action: &Value) -> Step {
        let mut s = self.env.step(action);
        self.elapsed += 1;
        if self.elapsed >= self.max_episode_steps {
            s.truncated = true;
        }
        s
    }
    fn render(&self) -> String {
        self.env.render()
    }
}

/// Add `episode_return` and `episode_length` to the info of each episode's
/// last step (`gymnasium.wrappers.RecordEpisodeStatistics`).
pub struct RecordEpisodeStatistics {
    env: Box<dyn Env>,
    ret: f64,
    len: usize,
}

impl RecordEpisodeStatistics {
    /// Wrap `env`.
    pub fn new(env: Box<dyn Env>) -> Self {
        RecordEpisodeStatistics { env, ret: 0.0, len: 0 }
    }
}

impl Env for RecordEpisodeStatistics {
    fn id(&self) -> &str {
        self.env.id()
    }
    fn observation_space(&self) -> &Space {
        self.env.observation_space()
    }
    fn action_space(&self) -> &Space {
        self.env.action_space()
    }
    fn action_space_mut(&mut self) -> &mut Space {
        self.env.action_space_mut()
    }
    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        (self.ret, self.len) = (0.0, 0);
        self.env.reset(seed)
    }
    fn step(&mut self, action: &Value) -> Step {
        let mut s = self.env.step(action);
        self.ret += s.reward;
        self.len += 1;
        if s.terminated || s.truncated {
            s.info.insert("episode_return".into(), self.ret);
            s.info.insert("episode_length".into(), self.len as f64);
        }
        s
    }
    fn render(&self) -> String {
        self.env.render()
    }
}

/// One step of a [`SyncVectorEnv`].
#[derive(Clone, Debug, PartialEq)]
pub struct VectorStep {
    /// Observation of each sub-environment.
    pub obs: Vec<Value>,
    /// Rewards.
    pub rewards: Vec<f64>,
    /// Terminated flags.
    pub terminated: Vec<bool>,
    /// Truncated flags.
    pub truncated: Vec<bool>,
}

/// Several environments stepped together (`gymnasium.vector.SyncVectorEnv`)
/// with Gymnasium 1.x's default "next step" autoreset: the step after an
/// episode ends resets that sub-environment (its action is ignored) and
/// returns the first observation with reward 0 and both flags false.
pub struct SyncVectorEnv {
    envs: Vec<Box<dyn Env>>,
    autoreset: Vec<bool>,
}

impl SyncVectorEnv {
    /// Step `envs` together.
    pub fn new(envs: Vec<Box<dyn Env>>) -> Self {
        let n = envs.len();
        SyncVectorEnv { envs, autoreset: vec![false; n] }
    }

    /// `n` copies of a registered environment.
    pub fn make(id: &str, n: usize) -> Result<Self, String> {
        Ok(Self::new((0..n).map(|_| super::make(id)).collect::<Result<_, _>>()?))
    }

    /// Number of sub-environments.
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// The sub-environments.
    pub fn envs_mut(&mut self) -> &mut [Box<dyn Env>] {
        &mut self.envs
    }

    /// Reset all; with a seed, sub-environment `i` gets `seed + i`.
    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Value> {
        self.autoreset.iter_mut().for_each(|a| *a = false);
        self.envs.iter_mut().enumerate().map(|(i, e)| e.reset(seed.map(|s| s + i as u64)).0).collect()
    }

    /// Step every sub-environment with its action.
    pub fn step(&mut self, actions: &[Value]) -> VectorStep {
        assert_eq!(actions.len(), self.envs.len(), "one action per sub-environment");
        let n = self.envs.len();
        let mut out = VectorStep { obs: Vec::with_capacity(n), rewards: vec![0.0; n], terminated: vec![false; n], truncated: vec![false; n] };
        for (i, (env, a)) in self.envs.iter_mut().zip(actions).enumerate() {
            if self.autoreset[i] {
                out.obs.push(env.reset(None).0);
            } else {
                let s = env.step(a);
                out.obs.push(s.obs);
                (out.rewards[i], out.terminated[i], out.truncated[i]) = (s.reward, s.terminated, s.truncated);
            }
            self.autoreset[i] = out.terminated[i] || out.truncated[i];
        }
        out
    }
}
