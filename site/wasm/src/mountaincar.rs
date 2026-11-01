//! Tabular Q-learning on MountainCar: the (position, velocity) plane cut
//! into a grid, one action value per cell and action, ε-greedy exploration.

use ruai::world::{self, Env, Value};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const LO: [f32; 2] = [-1.2, -0.07];
const HI: [f32; 2] = [0.6, 0.07];

struct Learner {
    bins: usize,
    alpha: f64,
    eps: f64,
    q: Vec<[f64; 3]>,
    env: Box<dyn Env>,
    rng: ruai::rng::Generator,
    episodes: usize,
    lengths: Vec<f64>,
}

thread_local! {
    static MC: RefCell<Option<Learner>> = const { RefCell::new(None) };
}

fn cell(bins: usize, obs: &[f32]) -> usize {
    let b = |i: usize| ((((obs[i] - LO[i]) / (HI[i] - LO[i])) * bins as f32) as usize).min(bins - 1);
    b(1) * bins + b(0)
}

fn best(q: &[f64; 3]) -> usize {
    (0..3).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap()
}

/// A new learner on a `bins` × `bins` grid with step size `alpha` and
/// initial exploration rate `eps` (all action values start at 0, which is
/// optimistic: every step is rewarded -1).
#[wasm_bindgen]
pub fn mc_reset(seed: u32, bins: usize, alpha: f64, eps: f64) {
    let mut env = world::make("MountainCar-v0").unwrap();
    env.reset(Some(seed as u64));
    let rng = ruai::rng::Generator::new(seed as u64 + 1);
    MC.with(|m| *m.borrow_mut() = Some(Learner { bins, alpha, eps, q: vec![[0.0; 3]; bins * bins], env, rng, episodes: 0, lengths: vec![] }));
}

/// Learn from `n` episodes: [episodes, steps of the last, mean steps of the last 50].
#[wasm_bindgen]
pub fn mc_train(n: usize) -> Vec<f64> {
    MC.with(|m| {
        let mut m = m.borrow_mut();
        let m = m.as_mut().expect("mc_reset first");
        for _ in 0..n {
            let eps = m.eps * 0.995f64.powi(m.episodes.min(10_000) as i32);
            let (obs, _) = m.env.reset(None);
            let mut s = cell(m.bins, obs.floats());
            let mut steps = 0;
            loop {
                let a = if m.rng.random() < eps { m.rng.integers(0, 3) as usize } else { best(&m.q[s]) };
                let r = m.env.step(&Value::Int(a as i64));
                let s2 = cell(m.bins, r.obs.floats());
                let target = r.reward + if r.terminated { 0.0 } else { 0.99 * m.q[s2][best(&m.q[s2])] };
                m.q[s][a] += m.alpha * (target - m.q[s][a]);
                s = s2;
                steps += 1;
                if r.terminated || r.truncated {
                    break;
                }
            }
            m.episodes += 1;
            m.lengths.push(steps as f64);
        }
        let k = m.lengths.len().min(50);
        let recent = m.lengths[m.lengths.len() - k..].iter().sum::<f64>() / k.max(1) as f64;
        vec![m.episodes as f64, m.lengths.last().copied().unwrap_or(0.0), recent]
    })
}

/// The grid size and max_a Q per cell (velocity rows from the lowest, positions in columns).
#[wasm_bindgen]
pub fn mc_values() -> Vec<f32> {
    MC.with(|m| {
        let m = m.borrow();
        let m = m.as_ref().expect("mc_reset first");
        let mut out = vec![m.bins as f32];
        out.extend(m.q.iter().map(|q| q[best(q)] as f32));
        out
    })
}

/// A greedy episode to animate: (position, velocity) per step.
#[wasm_bindgen]
pub fn mc_play() -> Vec<f32> {
    MC.with(|m| {
        let mut m = m.borrow_mut();
        let m = m.as_mut().expect("mc_reset first");
        let (obs, _) = m.env.reset(None);
        let mut obs = obs.floats().to_vec();
        let mut out = obs.clone();
        loop {
            let a = best(&m.q[cell(m.bins, &obs)]);
            let r = m.env.step(&Value::Int(a as i64));
            obs = r.obs.floats().to_vec();
            out.extend_from_slice(&obs);
            if r.terminated || r.truncated {
                break;
            }
        }
        out
    })
}
