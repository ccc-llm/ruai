//! REINFORCE on CartPole: a policy network (4 → 64 → 2) learns to balance
//! the pole from the returns of its own episodes.

use crate::mlp;
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use ruai::world::{CartPole, Env, Value};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const MAX_STEPS: usize = 500;
const GAMMA: f64 = 0.99;

struct Agent {
    policy: nn::Sequential,
    opt: Adam,
    env: CartPole,
    rng: ruai::rng::Generator,
    episodes: usize,
    returns: Vec<f64>,
}

thread_local! {
    static AGENT: RefCell<Option<Agent>> = const { RefCell::new(None) };
}

/// A new, untrained agent.
#[wasm_bindgen]
pub fn cp_reset(seed: u32) {
    ruai::rand::manual_seed(seed as u64);
    let policy = mlp(&[4, 64, 2], "tanh", Device::Cpu);
    let opt = Adam::new(policy.parameters(), 0.01);
    let mut env = CartPole::new();
    env.reset(Some(seed as u64));
    let rng = ruai::rng::Generator::new(seed as u64 + 1);
    AGENT.with(|a| *a.borrow_mut() = Some(Agent { policy, opt, env, rng, episodes: 0, returns: vec![] }));
}

/// P(push right) for an observation.
fn p_right(policy: &nn::Sequential, obs: &[f32]) -> f64 {
    let l = no_grad(|| policy.forward(&Tensor::from_vec(obs.to_vec(), &[1, 4], Device::Cpu))).to_vec();
    1.0 / (1.0 + ((l[0] - l[1]) as f64).exp())
}

/// One episode: observations, actions and rewards, sampling actions or
/// taking the likelier one (`greedy`).
fn rollout(a: &mut Agent, greedy: bool) -> (Vec<f32>, Vec<usize>, usize) {
    let (obs, _) = a.env.reset(None);
    let mut obs = obs.floats().to_vec();
    let (mut states, mut actions) = (vec![], vec![]);
    for _ in 0..MAX_STEPS {
        let p = p_right(&a.policy, &obs);
        let act = if greedy { usize::from(p > 0.5) } else { usize::from(a.rng.random() < p) };
        states.extend_from_slice(&obs);
        actions.push(act);
        let s = a.env.step(&Value::Int(act as i64));
        obs = s.obs.floats().to_vec();
        if s.terminated {
            break;
        }
    }
    let n = actions.len();
    (states, actions, n)
}

/// Train on `n` episodes (one policy-gradient step each): [episodes, last
/// return, mean return of the last 20].
#[wasm_bindgen]
pub fn cp_train(n: usize) -> Vec<f64> {
    AGENT.with(|a| {
        let mut a = a.borrow_mut();
        let a = a.as_mut().expect("cp_reset first");
        for _ in 0..n {
            let (states, actions, t) = rollout(a, false);
            // discounted returns, normalized
            let mut g = vec![0.0; t];
            let mut acc = 0.0;
            for i in (0..t).rev() {
                acc = 1.0 + GAMMA * acc;
                g[i] = acc;
            }
            let mean = g.iter().sum::<f64>() / t as f64;
            let std = (g.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / t as f64).sqrt().max(1e-6);
            // loss = -mean(G · log π(a|s)): one-hot weights over the log-probabilities
            let mut w = vec![0.0f32; 2 * t];
            for i in 0..t {
                w[2 * i + actions[i]] = ((g[i] - mean) / std) as f32;
            }
            a.opt.zero_grad();
            let logp = a.policy.forward(&Tensor::from_vec(states, &[t, 4], Device::Cpu)).log_softmax(-1);
            let loss = logp.mul(&Tensor::from_vec(w, &[t, 2], Device::Cpu)).sum_all().mul_scalar(-1.0 / t as f32);
            loss.backward();
            a.opt.step();
            a.episodes += 1;
            a.returns.push(t as f64);
        }
        let k = a.returns.len().min(20);
        let recent = a.returns[a.returns.len() - k..].iter().sum::<f64>() / k.max(1) as f64;
        vec![a.episodes as f64, a.returns.last().copied().unwrap_or(0.0), recent]
    })
}

/// The returns of all episodes so far.
#[wasm_bindgen]
pub fn cp_returns() -> Vec<f64> {
    AGENT.with(|a| a.borrow().as_ref().map(|a| a.returns.clone()).unwrap_or_default())
}

/// A greedy episode to animate: (cart x, pole angle) per step.
#[wasm_bindgen]
pub fn cp_play() -> Vec<f32> {
    AGENT.with(|a| {
        let mut a = a.borrow_mut();
        let a = a.as_mut().expect("cp_reset first");
        let (states, _, _) = rollout(a, true);
        states.chunks(4).flat_map(|s| [s[0], s[2]]).collect()
    })
}
