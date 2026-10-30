//! ResNet, ViT and padded max pooling. Set `RUSTNN_DEVICE=mps` to run on the GPU.

use rustnn::functional as F;
use rustnn::nn::resnet::{ResNet, ResNetConfig};
use rustnn::nn::vit::{ViT, ViTConfig};
use rustnn::nn::{self, Module};
use rustnn::optim::{Adam, Optimizer};
use rustnn::{no_grad, rand, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn max_pool_padding_is_negative_infinity() {
    let x = Tensor::from_vec((1..=9).map(|v| -(v as f32)).collect(), &[1, 1, 3, 3], dev());
    let y = F::max_pool2d_padded(&x, 3, 2, 1);
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    // zero padding would give 0 everywhere
    // windows: rows/cols {-1,0,1} and {1,2,3} of the padded image
    assert_eq!(y.to_vec(), vec![-1., -2., -4., -5.]);
    assert_eq!(nn::MaxPool2d::with_padding(3, 2, 1).forward(&x).to_vec(), y.to_vec());
}

#[test]
fn parameter_counts_match_reference_implementations() {
    let _l = lock();
    // torchvision.models.resnet{18,34,50}(num_classes=1000), timm vit_base_patch16_224
    assert_eq!(ResNet::new(ResNetConfig::resnet18(1000), dev()).num_parameters(), 11_689_512);
    assert_eq!(ResNet::new(ResNetConfig::resnet34(1000), dev()).num_parameters(), 21_797_672);
    assert_eq!(ResNet::new(ResNetConfig::resnet50(1000), dev()).num_parameters(), 25_557_032);
    assert_eq!(ViT::new(ViTConfig::base_16(1000), dev()).num_parameters(), 86_567_656);
}

#[test]
fn resnet_names_and_shapes() {
    let _l = lock();
    rand::manual_seed(0);
    let m = ResNet::new(ResNetConfig::resnet50(10), dev());
    let sd: std::collections::HashMap<String, Vec<usize>> =
        m.state_dict().into_iter().map(|(n, t)| (n, t.shape().to_vec())).collect();
    assert_eq!(sd["conv1.weight"], vec![64, 3, 7, 7]);
    assert_eq!(sd["layer1.0.conv3.weight"], vec![256, 64, 1, 1]);
    assert_eq!(sd["layer1.0.downsample.0.weight"], vec![256, 64, 1, 1]);
    assert_eq!(sd["layer2.0.conv2.weight"], vec![128, 128, 3, 3]);
    assert_eq!(sd["layer4.2.bn3.running_var"], vec![1, 2048, 1, 1]);
    assert_eq!(sd["fc.weight"], vec![2048, 10]);
    assert!(!sd.contains_key("layer1.1.downsample.0.weight"));

    let m = ResNet::new(ResNetConfig::resnet18(10), dev());
    m.eval();
    let x = Tensor::randn(&[2, 3, 64, 64], dev());
    no_grad(|| {
        assert_eq!(m.features(&x).shape(), &[2, 512, 2, 2]);
        assert_eq!(m.forward(&x).shape(), &[2, 10]);
    });
}

#[test]
fn resnet_loads_torchvision_layout() {
    let _l = lock();
    rand::manual_seed(1);
    let cfg = ResNetConfig::small(3, &[8, 16], 1, 5);
    let a = ResNet::new(cfg.clone(), dev());
    // give the batch-norm buffers non-trivial values, then evaluate
    let x = Tensor::randn(&[4, 3, 8, 8], dev());
    a.forward(&x);
    a.eval();
    // export the way torchvision lays it out: 1-D batch-norm tensors,
    // fc.weight as [out, in], plus num_batches_tracked
    let mut torch: Vec<(String, Tensor)> = Vec::new();
    for (n, t) in a.state_dict() {
        let t = if n == "fc.weight" {
            t.t()
        } else if t.rank() == 4 && t.shape()[0] == 1 {
            t.reshape(&[t.numel()])
        } else {
            t
        };
        if n.ends_with("running_var") {
            let prefix = n.trim_end_matches("running_var");
            torch.push((format!("{prefix}num_batches_tracked"), Tensor::from_vec_i32(vec![3], &[], dev())));
        }
        torch.push((n, t));
    }
    assert_eq!(torch.iter().find(|(n, _)| n == "fc.weight").unwrap().1.shape(), &[5, 16]);
    assert_eq!(torch.iter().find(|(n, _)| n == "bn1.weight").unwrap().1.shape(), &[8]);
    let b = ResNet::new(cfg, dev());
    b.load_torchvision_state_dict(torch).unwrap();
    b.eval();
    no_grad(|| assert!(max_abs_diff(&a.forward(&x).to_vec(), &b.forward(&x).to_vec()) < 1e-5));
}

/// 4 classes: which quadrant of an 8x8 image holds a bright 2x2 square.
fn quadrant_batch(n: usize) -> (Tensor, Vec<usize>) {
    let mut img = rand::normal_vec(n * 64, 0.0, 0.2);
    let mut labels = Vec::new();
    for i in 0..n {
        let q = rand::randint(4);
        let (r0, c0) = ((q / 2) * 4 + rand::randint(3), (q % 2) * 4 + rand::randint(3));
        for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            img[i * 64 + (r0 + dr) * 8 + c0 + dc] += 1.5;
        }
        labels.push(q);
    }
    (Tensor::from_vec(img, &[n, 1, 8, 8], dev()), labels)
}

fn train_classifier(model: &dyn Module, steps: usize, lr: f32) -> f32 {
    let mut opt = Adam::new(model.parameters(), lr);
    for _ in 0..steps {
        let (x, y) = quadrant_batch(32);
        opt.zero_grad();
        F::cross_entropy(&model.forward(&x), &y, 0.0).backward();
        opt.step();
    }
    model.eval();
    let (x, y) = quadrant_batch(200);
    no_grad(|| F::accuracy(&model.forward(&x), &y))
}

#[test]
fn small_resnet_learns() {
    let _l = lock();
    rand::manual_seed(2);
    let m = ResNet::new(ResNetConfig::small(1, &[8, 16], 1, 4), dev());
    let acc = train_classifier(&m, 60, 3e-3);
    assert!(acc > 0.9, "accuracy {acc}");
}

#[test]
fn small_vit_learns() {
    let _l = lock();
    rand::manual_seed(3);
    let m = ViT::new(ViTConfig::tiny(8, 4, 1, 4, 32, 2, 4), dev());
    assert_eq!(m.config.num_patches(), 4);
    let x = Tensor::randn(&[3, 1, 8, 8], dev());
    assert_eq!(m.encode(&x).shape(), &[3, 5, 32]);
    let acc = train_classifier(&m, 100, 3e-3);
    assert!(acc > 0.9, "accuracy {acc}");
}
