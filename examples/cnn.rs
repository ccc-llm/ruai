//! CNN image classification with a multi-threaded DataLoader, per-epoch
//! checkpoints and resumable training.
//!
//! Uses MNIST if the raw IDX files are in `data/mnist/`
//! (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
//! t10k-labels-idx1-ubyte); otherwise generates a synthetic shapes dataset.
//!
//!     cargo run --release --example cnn
//!     cargo run --release --example cnn -- --resume        # continue from runs/<model>/last.safetensors
//!     cargo run --release --features mps --example cnn -- --device mps
//!     cargo run --release --example cnn -- --model resnet  # small ResNet (cnn | resnet | vit)
//!     cargo run --release --example cnn -- --model vit     # Vision Transformer (4x4 patches)

use rustnn::checkpoint::{self, TrainState};
use rustnn::data::{Batch, ClassificationData, DataLoader};
use rustnn::metrics::{Accuracy, RunningMean, Throughput};
use rustnn::nn::resnet::{ResNet, ResNetConfig};
use rustnn::nn::vit::{ViT, ViTConfig};
use rustnn::nn::{self, Module, Sequential};
use rustnn::optim::lr::{CosineLr, LrScheduler};
use rustnn::optim::{Adam, Optimizer};
use rustnn::prelude::*;
use rustnn::{rand, serialize};
use std::path::Path;

/// `--model cnn|resnet|vit` (default cnn).
fn model_arg() -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == "--model").map_or("cnn".into(), |i| args[i + 1].clone())
}

/// `--device cpu|mps|cuda|auto` (or the `RUSTNN_DEVICE` env var); defaults to CPU.
fn device_arg() -> Device {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--device") {
        Some(i) => args.get(i + 1).expect("--device needs a value").parse().unwrap(),
        None => Device::from_env(),
    }
}

fn read_idx(path: &Path) -> std::io::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    let ndim = bytes[3] as usize;
    Ok(bytes[4 + 4 * ndim..].to_vec())
}

fn load_mnist(dir: &Path, prefix: &str) -> Option<ClassificationData> {
    let imgs = read_idx(&dir.join(format!("{prefix}-images-idx3-ubyte"))).ok()?;
    let labels = read_idx(&dir.join(format!("{prefix}-labels-idx1-ubyte"))).ok()?;
    let features = imgs.iter().map(|&p| (p as f32 / 255.0 - 0.1307) / 0.3081).collect();
    Some(ClassificationData::new(features, labels.iter().map(|&l| l as usize).collect(), &[1, 28, 28]))
}

/// 16x16 images of 4 shape classes at random positions, with noise:
/// 0 = horizontal bar, 1 = vertical bar, 2 = diagonal, 3 = hollow box.
fn synthetic(n: usize) -> ClassificationData {
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
    ClassificationData::new(images, labels, &[1, s, s])
}

fn build_model(kind: &str, size: usize, classes: usize, device: Device) -> Box<dyn Module> {
    match kind {
        "cnn" => Box::new(build_cnn(size, classes, device)),
        "resnet" => Box::new(ResNet::new(ResNetConfig::small(1, &[16, 32, 64], 1, classes), device)),
        "vit" => Box::new(ViT::new(ViTConfig::tiny(size, 4, 1, classes, 64, 4, 4), device)),
        other => panic!("--model expects cnn, resnet or vit, got {other}"),
    }
}

fn build_cnn(size: usize, classes: usize, device: Device) -> Sequential {
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

/// `[B, C*H*W]` batch inputs back to images.
fn images(b: &Batch, size: usize) -> Tensor {
    b.inputs.reshape(&[b.inputs.shape()[0], 1, size, size])
}

fn evaluate(model: &dyn Module, loader: &DataLoader<ClassificationData, Batch>, size: usize) -> f32 {
    model.eval();
    let mut acc = Accuracy::new();
    no_grad(|| {
        for b in loader.iter() {
            acc.update(&model.forward(&images(&b, size)), &b.targets);
        }
    });
    model.train();
    acc.value()
}

fn main() {
    rand::manual_seed(0);
    let device = device_arg();
    let resume = std::env::args().any(|a| a == "--resume");
    let kind = model_arg();
    let ckpt = format!("runs/{kind}/last.safetensors");
    println!("device: {device}, model: {kind}");
    let dir = Path::new("data/mnist");
    let (train, test, classes, epochs) = match (load_mnist(dir, "train"), load_mnist(dir, "t10k")) {
        (Some(a), Some(b)) => {
            println!("Using MNIST: {} train / {} test", a.labels.len(), b.labels.len());
            (a, b, 10, 2)
        }
        _ => {
            println!("MNIST not found in data/mnist, using synthetic shapes dataset");
            (synthetic(4000), synthetic(800), 4, 4)
        }
    };
    let size = train.sample_shape[1];

    // batches are assembled on 2 background threads while the model trains
    let train_loader = DataLoader::new(train, 64, ClassificationData::collate(device))
        .shuffle(true)
        .drop_last(true)
        .seed(1)
        .num_workers(2);
    let test_loader = DataLoader::new(test, 256, ClassificationData::collate(device));

    let model = build_model(&kind, size, classes, device);
    println!("model parameters: {}", model.num_parameters());
    let mut opt = Adam::new(model.parameters(), 2e-3);
    let total_steps = epochs * train_loader.len();
    let sched = CosineLr { base_lr: 2e-3, min_lr: 1e-4, warmup: 20, total: total_steps };

    let mut state = TrainState::default();
    if resume && Path::new(&ckpt).exists() {
        state = checkpoint::load(&ckpt, &model, &mut opt).expect("load checkpoint");
        println!("resumed from {ckpt}: epoch {}, step {}", state.epoch, state.step);
    }

    while (state.epoch as usize) < epochs {
        let mut loss_avg = RunningMean::new();
        let mut speed = Throughput::new();
        for (i, b) in train_loader.epoch(state.epoch).enumerate() {
            sched.apply(&mut opt, state.step as usize);
            opt.zero_grad();
            let loss = F::cross_entropy(&model.forward(&images(&b, size)), &b.targets, 0.0);
            loss.backward();
            opt.step();
            state.step += 1;
            loss_avg.add(loss.item());
            speed.add(b.targets.numel());
            if (i + 1) % 100 == 0 {
                println!("  epoch {} step {}/{} loss {:.4}", state.epoch + 1, i + 1, train_loader.len(), loss_avg.mean());
            }
        }
        state.epoch += 1;
        let acc = evaluate(&model, &test_loader, size);
        state.extra.insert("test_acc".into(), format!("{acc:.4}"));
        checkpoint::save(&ckpt, &model, &opt, &state).expect("save checkpoint");
        println!(
            "epoch {}: train loss {:.4}, test acc {:.2}%, {:.0} img/s, lr {:.5}  (checkpoint: {ckpt})",
            state.epoch,
            loss_avg.mean(),
            100.0 * acc,
            speed.per_sec(),
            opt.lr()
        );
    }

    // export inference weights (safetensors: loadable from PyTorch / HF too), reload, evaluate
    let path = format!("{kind}.safetensors");
    serialize::save(&path, &model).expect("save");
    let restored = build_model(&kind, size, classes, device);
    serialize::load(&path, &restored).expect("load");
    println!("reloaded model test acc: {:.2}% (weights in {path})", 100.0 * evaluate(&*restored, &test_loader, size));
}
