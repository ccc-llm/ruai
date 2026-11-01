//! Neural network playground: a small MLP learns to separate two classes of
//! 2-D points (circle, xor, spiral, blobs, moons), trained full-batch with Adam.

use crate::{mlp, normal};
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use std::cell::RefCell;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

struct Playground {
    model: nn::Sequential,
    opt: Adam,
    x: Tensor,
    y: Vec<usize>,
    step: usize,
}

thread_local! {
    static PG: RefCell<Option<Playground>> = const { RefCell::new(None) };
}

/// `n` points of `dataset` in [-1, 1]²: (x, y, label) triples.
fn points(dataset: &str, n: usize, noise: f64, seed: u64) -> Vec<f32> {
    let mut g = ruai::rng::Generator::new(seed);
    let mut out = Vec::with_capacity(3 * n);
    for i in 0..n {
        let label = i % 2;
        let (x, y) = match dataset {
            "xor" => {
                let mut x = g.uniform(-1.0, 1.0);
                let mut y = g.uniform(-1.0, 1.0);
                // keep a margin around the axes
                x += 0.08 * x.signum();
                y += 0.08 * y.signum();
                let l = usize::from(x * y > 0.0);
                out.extend_from_slice(&[(x * 0.9) as f32, (y * 0.9) as f32, l as f32]);
                continue;
            }
            "spiral" => {
                let t = g.random();
                let r = 0.08 + 0.85 * t;
                let a = 3.5 * PI * t + label as f64 * PI;
                (r * a.cos(), r * a.sin())
            }
            "blobs" => {
                let c = if label == 0 { -0.45 } else { 0.45 };
                (c + 0.22 * normal(&mut g), c + 0.22 * normal(&mut g))
            }
            "moons" => {
                let a = g.uniform(0.0, PI);
                if label == 0 {
                    (a.cos() * 0.6 - 0.3, a.sin() * 0.6 - 0.15)
                } else {
                    (0.3 - a.cos() * 0.6, 0.15 - a.sin() * 0.6)
                }
            }
            // circle
            _ => {
                let a = g.uniform(0.0, 2.0 * PI);
                let r = if label == 0 { g.uniform(0.0, 0.45) } else { g.uniform(0.65, 0.95) };
                (r * a.cos(), r * a.sin())
            }
        };
        let x = (x + noise * normal(&mut g)).clamp(-1.0, 1.0);
        let y = (y + noise * normal(&mut g)).clamp(-1.0, 1.0);
        out.extend_from_slice(&[x as f32, y as f32, label as f32]);
    }
    out
}

/// A new problem and model: the points as (x, y, label) triples.
#[wasm_bindgen]
pub fn pg_reset(dataset: &str, n: usize, noise: f64, hidden: Vec<u32>, act: &str, lr: f32, seed: u32) -> Vec<f32> {
    ruai::rand::manual_seed(seed as u64);
    let pts = points(dataset, n, noise, seed as u64);
    let mut xs = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for p in pts.chunks(3) {
        xs.extend_from_slice(&[p[0], p[1]]);
        y.push(p[2] as usize);
    }
    let mut sizes = vec![2];
    sizes.extend(hidden.iter().map(|&h| h as usize));
    sizes.push(2);
    let model = mlp(&sizes, act, Device::Cpu);
    let opt = Adam::new(model.parameters(), lr);
    let x = Tensor::from_vec(xs, &[n, 2], Device::Cpu);
    PG.with(|p| *p.borrow_mut() = Some(Playground { model, opt, x, y, step: 0 }));
    pts
}

/// `steps` full-batch Adam steps: [step, loss, accuracy].
#[wasm_bindgen]
pub fn pg_train(steps: usize) -> Vec<f64> {
    PG.with(|p| {
        let mut p = p.borrow_mut();
        let p = p.as_mut().expect("pg_reset first");
        let mut loss = 0.0;
        for _ in 0..steps {
            p.opt.zero_grad();
            let logits = p.model.forward(&p.x);
            let l = F::cross_entropy(&logits, &p.y, 0.0);
            l.backward();
            p.opt.step();
            loss = l.item() as f64;
            p.step += 1;
        }
        let logits = no_grad(|| p.model.forward(&p.x)).to_vec();
        let correct = p.y.iter().enumerate().filter(|&(i, &y)| usize::from(logits[2 * i + 1] > logits[2 * i]) == y).count();
        vec![p.step as f64, loss, correct as f64 / p.y.len() as f64]
    })
}

/// P(class 1) on a `res` × `res` grid over [-1, 1]², top row first.
#[wasm_bindgen]
pub fn pg_grid(res: usize) -> Vec<f32> {
    PG.with(|p| {
        let p = p.borrow();
        let p = p.as_ref().expect("pg_reset first");
        let mut xs = Vec::with_capacity(2 * res * res);
        for r in 0..res {
            for c in 0..res {
                let x = -1.0 + 2.0 * (c as f32 + 0.5) / res as f32;
                let y = 1.0 - 2.0 * (r as f32 + 0.5) / res as f32;
                xs.extend_from_slice(&[x, y]);
            }
        }
        let logits = no_grad(|| p.model.forward(&Tensor::from_vec(xs, &[res * res, 2], Device::Cpu))).to_vec();
        logits.chunks(2).map(|l| 1.0 / (1.0 + (l[0] - l[1]).exp())).collect()
    })
}
