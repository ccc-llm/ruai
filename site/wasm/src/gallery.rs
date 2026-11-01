//! A gallery of ruai::plot figures (matplotlib's API, drawn on a canvas).

use crate::normal;
use ruai::plot::{Figure, Style};
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

/// The names of the figures.
#[wasm_bindgen]
pub fn gallery_names() -> Vec<String> {
    ["lines", "scatter", "histogram", "bars", "mandelbrot", "subplots", "lissajous"].map(String::from).to_vec()
}

/// Figure `name` at `width` × `height` pixels: JSON for `ruaiDraw`.
#[wasm_bindgen]
pub fn gallery(name: &str, width: f64, height: f64, seed: u32) -> String {
    let mut fig = Figure::new(width / 100.0, height / 100.0);
    let mut g = ruai::rng::Generator::new(seed as u64);
    match name {
        "scatter" => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            for (k, (cx, cy)) in [(-2.0, 0.0), (1.5, 1.5), (1.0, -2.0)].iter().enumerate() {
                let x: Vec<f64> = (0..150).map(|_| cx + 0.8 * normal(&mut g)).collect();
                let y: Vec<f64> = (0..150).map(|_| cy + 0.8 * normal(&mut g)).collect();
                ax.scatter(x, y, 16.0, 0.6, &Style::default().label(&format!("cluster {k}")));
            }
            ax.title = "Three Gaussian clusters".into();
            ax.legend = true;
        }
        "histogram" => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            let a: Vec<f64> = (0..4000).map(|_| normal(&mut g)).collect();
            let b: Vec<f64> = (0..4000).map(|_| 2.5 + 0.6 * normal(&mut g)).collect();
            ax.hist(&a, 40, "C0", "N(0, 1)");
            ax.hist(&b, 40, "C1", "N(2.5, 0.6²)");
            ax.title = "Histograms of normal samples".into();
            ax.legend = true;
        }
        "bars" => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            let names: Vec<String> = ["Rust", "py0", "WebGPU", "Metal", "CUDA"].map(String::from).to_vec();
            let pos = ax.category_positions(&names);
            ax.bar(pos, vec![9.0, 7.5, 8.0, 6.5, 7.0], 0.6, "C2", "score");
            ax.title = "Bar chart with categories".into();
            ax.grid_lines = true;
        }
        "mandelbrot" => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            let (rows, cols) = (120, 180);
            let mut v = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    let (x0, y0) = (-2.2 + 3.0 * c as f64 / cols as f64, -1.0 + 2.0 * r as f64 / rows as f64);
                    let (mut x, mut y, mut i) = (0.0, 0.0, 0);
                    while i < 60 && x * x + y * y < 4.0 {
                        (x, y) = (x * x - y * y + x0, 2.0 * x * y + y0);
                        i += 1;
                    }
                    v.push((i as f64 + 1.0).ln());
                }
            }
            ax.imshow(v, rows, cols, "viridis");
            ax.title = "The Mandelbrot set (imshow)".into();
        }
        "subplots" => {
            let t = linspace(0.0, 4.0 * PI, 300);
            for (i, (title, f)) in [
                ("damped", (|t: f64| (-0.2 * t).exp() * (3.0 * t).cos()) as fn(f64) -> f64),
                ("beats", |t: f64| (5.0 * t).sin() + (5.5 * t).sin()),
                ("square-ish", |t: f64| (1..8).step_by(2).map(|k| (k as f64 * t).sin() / k as f64).sum()),
                ("sawtooth-ish", |t: f64| (1..8).map(|k| -((k as f64 * t).sin()) / k as f64 * if k % 2 == 0 { -1.0 } else { 1.0 }).sum()),
            ]
            .iter()
            .enumerate()
            {
                let k = fig.add_subplot(2, 2, i + 1);
            let ax = &mut fig.axes[k];
                ax.plot(t.clone(), t.iter().map(|&x| f(x)).collect(), &Style::default().color(&format!("C{i}")));
                ax.title = title.to_string();
                ax.grid_lines = true;
            }
        }
        "lissajous" => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            let t = linspace(0.0, 2.0 * PI, 800);
            for (a, b) in [(3.0, 2.0), (5.0, 4.0)] {
                ax.plot(t.iter().map(|&t| (a * t + PI / 2.0).sin()).collect(), t.iter().map(|&t| (b * t).sin()).collect(), &Style::default().label(&format!("{a}:{b}")));
            }
            ax.title = "Lissajous curves".into();
            ax.legend = true;
        }
        // lines
        _ => {
            let k = fig.add_subplot(1, 1, 1);
            let ax = &mut fig.axes[k];
            let x = linspace(-PI, PI, 200);
            ax.plot(x.clone(), x.iter().map(|v| v.sin()).collect(), &Style::default().label("sin"));
            ax.plot(x.clone(), x.iter().map(|v| v.cos()).collect(), &Style::default().label("cos"));
            ax.plot(x.clone(), x.iter().map(|v| v.tanh()).collect(), &Style::default().linestyle("--").label("tanh"));
            ax.plot(x.clone(), x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(), &Style::default().linestyle(":").label("sigmoid"));
            ax.title = "Activation-style functions".into();
            ax.xlabel = "x".into();
            ax.legend = true;
            ax.grid_lines = true;
        }
    }
    ruai::plot::render::canvas_json(&fig)
}
