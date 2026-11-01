//! Optimizer race: SGD, SGD with momentum and Adam (ruai's optimizers, with
//! gradients from autograd) descend a 2-D loss surface from the same start.

use ruai::optim::{Adam, Optimizer, SGD};
use ruai::prelude::*;
use wasm_bindgen::prelude::*;

/// The loss at `p` = [x, y], built from tensor operations so autograd differentiates it.
fn loss(func: &str, p: &Tensor) -> Tensor {
    loss_rows(func, &p.reshape(&[2, 1])).sum_all()
}

/// The plotted region: [x0, x1, y0, y1].
#[wasm_bindgen]
pub fn descent_domain(func: &str) -> Vec<f64> {
    match func {
        "rosenbrock" => vec![-2.0, 2.0, -1.0, 3.0],
        "himmelblau" => vec![-5.0, 5.0, -5.0, 5.0],
        "saddle" => vec![-2.5, 2.5, -2.5, 2.5],
        _ => vec![-4.0, 4.0, -2.0, 2.0],
    }
}

/// log(1 + loss) on a `res` × `res` grid over the domain, top row first.
#[wasm_bindgen]
pub fn descent_surface(func: &str, res: usize) -> Vec<f32> {
    let d = descent_domain(func);
    let mut xs = Vec::with_capacity(res * res);
    let mut ys = Vec::with_capacity(res * res);
    for r in 0..res {
        for c in 0..res {
            xs.push((d[0] + (d[1] - d[0]) * (c as f64 + 0.5) / res as f64) as f32);
            ys.push((d[3] - (d[3] - d[2]) * (r as f64 + 0.5) / res as f64) as f32);
        }
    }
    // the same formulas on whole grids
    let (xs, ys) = (Tensor::from_vec(xs, &[res * res], Device::Cpu), Tensor::from_vec(ys, &[res * res], Device::Cpu));
    let p = Tensor::stack(&[&xs, &ys], 0);
    let v = no_grad(|| loss_rows(func, &p)).to_vec();
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    v.iter().map(|&v| (1.0 + v - lo).ln()).collect()
}

/// The loss at each column of a [2, n] batch: the n values.
fn loss_rows(func: &str, p: &Tensor) -> Tensor {
    let n = p.shape()[1];
    let x = p.narrow(0, 0, 1);
    let y = p.narrow(0, 1, 1);
    let v = match func {
        // (1 - x)² + 100 (y - x²)²
        "rosenbrock" => x.mul_scalar(-1.0).add_scalar(1.0).powf(2.0).add(&y.sub(&x.powf(2.0)).powf(2.0).mul_scalar(100.0)),
        // (x² + y - 11)² + (x + y² - 7)²
        "himmelblau" => x.powf(2.0).add(&y).add_scalar(-11.0).powf(2.0).add(&x.add(&y.powf(2.0)).add_scalar(-7.0).powf(2.0)),
        // x² - y² + y⁴ / 4 (a saddle at the origin, minima at y = ±√2)
        "saddle" => x.powf(2.0).sub(&y.powf(2.0)).add(&y.powf(4.0).mul_scalar(0.25)),
        // 0.5 x² + 10 y² (a long narrow valley)
        _ => x.powf(2.0).mul_scalar(0.5).add(&y.powf(2.0).mul_scalar(10.0)),
    };
    v.reshape(&[n])
}

/// The learning rates of SGD, momentum and Adam that suit the surface.
fn rates(func: &str) -> [f32; 3] {
    match func {
        "rosenbrock" => [0.0015, 0.0008, 0.1],
        "himmelblau" => [0.01, 0.002, 0.1],
        "saddle" => [0.03, 0.005, 0.05],
        _ => [0.09, 0.012, 0.15],
    }
}

/// The paths of SGD, momentum and Adam from (x, y), `steps` steps each with
/// the learning rates scaled by `scale`: 3 × (steps + 1) points as x, y pairs
/// (NaN after a path diverges).
#[wasm_bindgen]
pub fn descent_paths(func: &str, x: f32, y: f32, steps: usize, scale: f32) -> Vec<f32> {
    let lr = rates(func);
    let mut out = Vec::with_capacity(3 * 2 * (steps + 1));
    for (i, &lr) in lr.iter().enumerate() {
        let p = Tensor::from_vec(vec![x, y], &[2], Device::Cpu).requires_grad_();
        let lr = lr * scale;
        let mut opt: Box<dyn Optimizer> = match i {
            0 => Box::new(SGD::new(vec![p.clone()], lr)),
            1 => Box::new(SGD::with_momentum(vec![p.clone()], lr, 0.9, 0.0)),
            _ => Box::new(Adam::new(vec![p.clone()], lr)),
        };
        out.extend_from_slice(&[x, y]);
        let mut alive = true;
        for _ in 0..steps {
            if alive {
                opt.zero_grad();
                loss(func, &p).backward();
                opt.step();
                let v = p.to_vec();
                alive = v.iter().all(|c| c.is_finite() && c.abs() < 1e3);
                if alive {
                    out.extend_from_slice(&v);
                    continue;
                }
            }
            out.extend_from_slice(&[f32::NAN, f32::NAN]);
        }
    }
    out
}
