//! CNN image classification.
//!
//! Uses MNIST if the raw IDX files are in `data/mnist/`
//! (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
//! t10k-labels-idx1-ubyte); otherwise generates a synthetic shapes dataset.
//!
//!     cargo run --release --example cnn

use rustnn::nn::{self, Module, Sequential};
use rustnn::optim::{Adam, Optimizer};
use rustnn::prelude::*;
use rustnn::{rand, serialize};
use std::path::Path;
use std::time::Instant;

struct Dataset {
    images: Vec<f32>, // [N, 1, S, S]
    labels: Vec<usize>,
    size: usize,
    classes: usize,
}

impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn batch(&self, idx: &[usize], device: Device) -> (Tensor, Vec<usize>) {
        let px = self.size * self.size;
        let mut x = Vec::with_capacity(idx.len() * px);
        for &i in idx {
            x.extend_from_slice(&self.images[i * px..(i + 1) * px]);
        }
        let t = Tensor::from_vec(x, &[idx.len(), 1, self.size, self.size], device);
        (t, idx.iter().map(|&i| self.labels[i]).collect())
    }
}

fn read_idx(path: &Path) -> std::io::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    let ndim = bytes[3] as usize;
    Ok(bytes[4 + 4 * ndim..].to_vec())
}

fn load_mnist(dir: &Path, prefix: &str, limit: usize) -> Option<Dataset> {
    let imgs = read_idx(&dir.join(format!("{prefix}-images-idx3-ubyte"))).ok()?;
    let labels = read_idx(&dir.join(format!("{prefix}-labels-idx1-ubyte"))).ok()?;
    let n = labels.len().min(limit);
    Some(Dataset {
        images: imgs[..n * 784].iter().map(|&p| (p as f32 / 255.0 - 0.1307) / 0.3081).collect(),
        labels: labels[..n].iter().map(|&l| l as usize).collect(),
        size: 28,
        classes: 10,
    })
}

/// 16x16 images of 4 shape classes at random positions, with noise:
/// 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = hollow box.
fn synthetic(n: usize) -> Dataset {
    let s = 16;
    let mut images = vec![0.0; n * s * s];
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let c = rand::randint(4);
        let img = &mut images[i * s * s..(i + 1) * s * s];
        let (r0, c0) = (2 + rand::randint(6), 2 + rand::randint(6));
        let len = 6 + rand::randint(3);
        for k in 0..len {
            match c {
                0 => img[r0 * s + c0 + k] = 1.0,
                1 => img[(r0 + k) * s + c0] = 1.0,
                2 => img[(r0 + k) * s + c0 + k] = 1.0,
                _ => {
                    let e = len - 1;
                    for (y, x) in [(0, k), (e, k), (k, 0), (k, e)] {
                        img[(r0 + y) * s + c0 + x] = 1.0;
                    }
                }
            }
        }
        for (p, noise) in img.iter_mut().zip(rand::normal_vec(s * s, 0.0, 0.3)) {
            *p += noise;
        }
        labels.push(c);
    }
    Dataset { images, labels, size: s, classes: 4 }
}

fn build_model(size: usize, classes: usize, device: Device) -> Sequential {
    let flat = 16 * (size / 4) * (size / 4);
    Sequential::new()
        .add(nn::Conv2d::new(1, 8, 3, 1, 1, device))
        .add(nn::BatchNorm2d::new(8, device))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Conv2d::new(8, 16, 3, 1, 1, device))
        .add(nn::BatchNorm2d::new(16, device))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Flatten)
        .add(nn::Linear::new(flat, 64, device))
        .add(nn::ReLU)
        .add(nn::Dropout::new(0.2))
        .add(nn::Linear::new(64, classes, device))
}

fn evaluate(model: &Sequential, data: &Dataset, device: Device) -> f32 {
    model.eval();
    let mut correct = 0.0;
    no_grad(|| {
        for chunk in (0..data.len()).collect::<Vec<_>>().chunks(256) {
            let (x, y) = data.batch(chunk, device);
            correct += F::accuracy(&model.forward(&x), &y) * chunk.len() as f32;
        }
    });
    model.train();
    correct / data.len() as f32
}

fn main() {
    rand::manual_seed(0);
    let device = Device::Cpu;
    let dir = Path::new("data/mnist");
    let (train, test, epochs) = match (load_mnist(dir, "train", 60000), load_mnist(dir, "t10k", 10000)) {
        (Some(a), Some(b)) => {
            println!("Using MNIST: {} train / {} test", a.len(), b.len());
            (a, b, 2)
        }
        _ => {
            println!("MNIST not found in data/mnist, using synthetic shapes dataset");
            (synthetic(4000), synthetic(800), 4)
        }
    };

    let model = build_model(train.size, train.classes, device);
    println!("model parameters: {}", model.num_parameters());
    let mut opt = Adam::new(model.parameters(), 2e-3);
    let batch = 64;

    for epoch in 0..epochs {
        let t0 = Instant::now();
        let mut order: Vec<usize> = (0..train.len()).collect();
        rand::shuffle(&mut order);
        let mut total = 0.0;
        let steps = order.len() / batch;
        for (step, idx) in order.chunks_exact(batch).enumerate() {
            let (x, y) = train.batch(idx, device);
            opt.zero_grad();
            let loss = F::cross_entropy(&model.forward(&x), &y, 0.0);
            loss.backward();
            opt.step();
            total += loss.item();
            if (step + 1) % 100 == 0 {
                println!("  epoch {} step {}/{} loss {:.4}", epoch + 1, step + 1, steps, total / (step + 1) as f32);
            }
        }
        println!(
            "epoch {}: train loss {:.4}, test acc {:.2}%  ({:.1}s)",
            epoch + 1,
            total / steps as f32,
            100.0 * evaluate(&model, &test, device),
            t0.elapsed().as_secs_f32()
        );
    }

    // save, reload into a fresh model, and run inference
    let path = "cnn.rsnn";
    serialize::save(path, &model).expect("save");
    let restored = build_model(train.size, train.classes, device);
    serialize::load(path, &restored).expect("load");
    println!("reloaded model test acc: {:.2}% (weights in {path})", 100.0 * evaluate(&restored, &test, device));
}
