//! End-to-end checks: models actually learn, save/load round-trips.

use rustnn::functional as F;
use rustnn::nn::{self, Module, Sequential};
use rustnn::optim::{Adam, Optimizer, SGD};
use rustnn::{rand, serialize, Device, Tensor};

/// Device under test: set `RUSTNN_DEVICE=mps` to run this file on the GPU.
fn dev() -> Device {
    Device::from_env()
}

#[test]
fn mlp_learns_xor() {
    rand::manual_seed(3);
    let model = Sequential::new().add(nn::Linear::new(2, 16, dev())).add(nn::Tanh).add(nn::Linear::new(16, 2, dev()));
    let x = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]).to(dev());
    let y = [0usize, 1, 1, 0];
    let mut opt = SGD::with_momentum(model.parameters(), 0.1, 0.9, 0.0);
    for _ in 0..300 {
        opt.zero_grad();
        let loss = F::cross_entropy(&model.forward(&x), &y, 0.0);
        loss.backward();
        opt.step();
    }
    assert_eq!(F::accuracy(&model.forward(&x), &y), 1.0);
}

#[test]
fn small_cnn_overfits_batch() {
    rand::manual_seed(4);
    let model = Sequential::new()
        .add(nn::Conv2d::new(1, 4, 3, 1, 1, dev()))
        .add(nn::BatchNorm2d::new(4, dev()))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Flatten)
        .add(nn::Linear::new(4 * 4 * 4, 3, dev()));
    let x = Tensor::randn(&[6, 1, 8, 8], dev());
    let y = [0usize, 1, 2, 0, 1, 2];
    let mut opt = Adam::new(model.parameters(), 0.01);
    let mut last = f32::MAX;
    for _ in 0..60 {
        opt.zero_grad();
        let loss = F::cross_entropy(&model.forward(&x), &y, 0.0);
        loss.backward();
        opt.step();
        last = loss.item();
    }
    assert!(last < 0.1, "final loss {last}");
}

#[test]
fn save_and_load_roundtrip() {
    rand::manual_seed(5);
    let make = || Sequential::new().add(nn::Linear::new(3, 4, dev())).add(nn::BatchNorm2d::new(2, dev()));
    let a = make();
    let b = make();
    let path = std::env::temp_dir().join(format!("rustnn_test_{}.bin", std::process::id()));
    serialize::save(&path, &a).unwrap();
    serialize::load(&path, &b).unwrap();
    std::fs::remove_file(&path).ok();
    for ((na, ta), (nb, tb)) in a.state_dict().iter().zip(b.state_dict().iter()) {
        assert_eq!(na, nb);
        assert_eq!(ta.to_vec(), tb.to_vec(), "{na}");
    }
}

#[test]
fn gpt_memorizes_sequence() {
    rand::manual_seed(6);
    let cfg = nn::GPTConfig { vocab_size: 5, max_len: 8, dim: 16, n_heads: 2, n_layers: 1, dropout: 0.0 };
    let model = nn::GPT::new(cfg, dev());
    let seq = [0usize, 1, 2, 3, 4, 0, 1, 2, 3];
    let (inp, tgt) = (&seq[..8], &seq[1..]);
    let mut opt = Adam::new(model.parameters(), 0.01);
    for _ in 0..150 {
        opt.zero_grad();
        let logits = model.forward_ids(inp, 1, 8).reshape(&[8, 5]);
        F::cross_entropy(&logits, tgt, 0.0).backward();
        opt.step();
    }
    model.eval();
    assert_eq!(model.generate(&[0], 7, 0.0), vec![0, 1, 2, 3, 4, 0, 1, 2]);
}
