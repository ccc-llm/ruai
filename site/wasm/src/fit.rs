//! Curve fitting: an MLP learns a 1-D function from noisy samples; the page
//! draws the fit and the loss with ruai::plot.

use crate::{mlp, normal};
use ruai::optim::{Adam, Optimizer};
use ruai::plot::{Figure, Style};
use ruai::prelude::*;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

struct Fit {
    func: String,
    model: nn::Sequential,
    opt: Adam,
    xs: Vec<f32>,
    ys: Vec<f32>,
    x: Tensor,
    y: Tensor,
    steps: Vec<f64>,
    losses: Vec<f64>,
}

thread_local! {
    static FIT: RefCell<Option<Fit>> = const { RefCell::new(None) };
}

const X0: f64 = -3.0;
const X1: f64 = 3.0;

fn target(func: &str, x: f64) -> f64 {
    match func {
        "abs" => x.abs() - 1.5,
        "step" => if x < -1.0 { -1.0 } else if x < 1.0 { 0.5 } else { -0.5 },
        "sinc" => if x == 0.0 { 2.0 } else { 2.0 * (3.0 * x).sin() / (3.0 * x) },
        "cubic" => 0.15 * x * x * x - 0.8 * x,
        _ => x.sin() * 1.5,
    }
}

/// New samples of `func` and a new model.
#[wasm_bindgen]
pub fn fit_reset(func: &str, n: usize, noise: f64, act: &str, width: usize, seed: u32) {
    ruai::rand::manual_seed(seed as u64);
    let mut g = ruai::rng::Generator::new(seed as u64);
    let xs: Vec<f32> = (0..n).map(|_| g.uniform(X0, X1) as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| (target(func, x as f64) + noise * normal(&mut g)) as f32).collect();
    let model = mlp(&[1, width, width, 1], act, Device::Cpu);
    let opt = Adam::new(model.parameters(), 0.01);
    let x = Tensor::from_vec(xs.clone(), &[n, 1], Device::Cpu);
    let y = Tensor::from_vec(ys.clone(), &[n, 1], Device::Cpu);
    let fit = Fit { func: func.into(), model, opt, xs, ys, x, y, steps: vec![], losses: vec![] };
    FIT.with(|f| *f.borrow_mut() = Some(fit));
}

/// `steps` Adam steps on the mean squared error: [step, loss].
#[wasm_bindgen]
pub fn fit_train(steps: usize) -> Vec<f64> {
    FIT.with(|f| {
        let mut f = f.borrow_mut();
        let f = f.as_mut().expect("fit_reset first");
        let mut loss = f64::NAN;
        for _ in 0..steps {
            f.opt.zero_grad();
            let l = F::mse_loss(&f.model.forward(&f.x), &f.y);
            l.backward();
            f.opt.step();
            loss = l.item() as f64;
        }
        let step = f.steps.last().copied().unwrap_or(0.0) + steps as f64;
        if steps > 0 {
            f.steps.push(step);
            f.losses.push(loss.max(1e-6).log10());
        }
        vec![step, loss]
    })
}

/// The data, the true function and the model's fit, and the loss curve: JSON for `ruaiDraw`.
#[wasm_bindgen]
pub fn fit_plot(width: f64, height: f64) -> String {
    FIT.with(|f| {
        let f = f.borrow();
        let f = f.as_ref().expect("fit_reset first");
        let grid: Vec<f64> = (0..200).map(|i| X0 + (X1 - X0) * i as f64 / 199.0).collect();
        let pred = no_grad(|| f.model.forward(&Tensor::from_vec(grid.iter().map(|&x| x as f32).collect(), &[200, 1], Device::Cpu))).to_vec();
        let mut fig = Figure::new(width / 100.0, height / 100.0);
        let a = fig.add_subplot(1, 2, 1);
        let ax = &mut fig.axes[a];
        ax.scatter(f.xs.iter().map(|&x| x as f64).collect(), f.ys.iter().map(|&y| y as f64).collect(), 14.0, 0.6, &Style::default().label("samples"));
        ax.plot(grid.clone(), grid.iter().map(|&x| target(&f.func, x)).collect(), &Style::default().linestyle("--").color("gray").label("true function"));
        ax.plot(grid, pred.iter().map(|&p| p as f64).collect(), &Style::default().color("C3").label("MLP"));
        ax.title = "fit".into();
        ax.xlabel = "x".into();
        ax.legend = true;
        ax.grid_lines = true;
        let b = fig.add_subplot(1, 2, 2);
        let ax = &mut fig.axes[b];
        ax.plot(f.steps.clone(), f.losses.clone(), &Style::default().color("C1").label("log10 MSE"));
        ax.title = "loss".into();
        ax.xlabel = "step".into();
        ax.ylabel = "log10 MSE".into();
        ax.grid_lines = true;
        ruai::plot::render::canvas_json(&fig)
    })
}
