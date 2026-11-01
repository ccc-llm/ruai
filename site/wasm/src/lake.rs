//! FrozenLake by value iteration: the Bellman update swept over the lake's
//! model (its transition probabilities), then the greedy policy walked on
//! the slippery ice.

use ruai::world::{Env, FrozenLake, Value};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const GAMMA: f64 = 0.95;

struct Lake {
    env: FrozenLake,
    n: usize,
    v: Vec<f64>,
    sweeps: usize,
}

thread_local! {
    static LAKE: RefCell<Option<Lake>> = const { RefCell::new(None) };
}

/// The lake of `rows` (S, F, H, G joined by newlines), values reset to 0.
#[wasm_bindgen]
pub fn lake_new(rows: &str, slippery: bool, seed: u32) {
    let rows: Vec<&str> = rows.lines().collect();
    let mut env = FrozenLake::from_desc(&rows, slippery);
    env.reset(Some(seed as u64));
    let n = rows.len() * rows[0].len();
    LAKE.with(|l| *l.borrow_mut() = Some(Lake { env, n, v: vec![0.0; n], sweeps: 0 }));
}

fn q(l: &Lake, s: usize, a: usize) -> f64 {
    l.env.transitions(s, a).iter().map(|&(p, s2, r, done)| p * (r + if done { 0.0 } else { GAMMA * l.v[s2] })).sum()
}

fn policy(l: &Lake, s: usize) -> usize {
    (0..4).max_by(|&a, &b| q(l, s, a).total_cmp(&q(l, s, b))).unwrap()
}

/// One sweep of value iteration: [largest change, sweeps, values…, greedy actions…].
#[wasm_bindgen]
pub fn lake_sweep() -> Vec<f64> {
    LAKE.with(|l| {
        let mut l = l.borrow_mut();
        let l = l.as_mut().expect("lake_new first");
        let v: Vec<f64> = (0..l.n).map(|s| (0..4).map(|a| q(l, s, a)).fold(f64::MIN, f64::max)).collect();
        let delta = v.iter().zip(&l.v).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        l.v = v;
        l.sweeps += 1;
        let mut out = vec![delta, l.sweeps as f64];
        out.extend_from_slice(&l.v);
        out.extend((0..l.n).map(|s| policy(l, s) as f64));
        out
    })
}

/// Walk the greedy policy for one episode: the states visited.
#[wasm_bindgen]
pub fn lake_walk() -> Vec<u32> {
    LAKE.with(|l| {
        let mut l = l.borrow_mut();
        let l = l.as_mut().expect("lake_new first");
        let (s, _) = l.env.reset(None);
        let mut s = s.int() as usize;
        let mut path = vec![s as u32];
        for _ in 0..200 {
            let a = policy(l, s);
            let r = l.env.step(&Value::Int(a as i64));
            s = r.obs.int() as usize;
            path.push(s as u32);
            if r.terminated {
                break;
            }
        }
        path
    })
}
