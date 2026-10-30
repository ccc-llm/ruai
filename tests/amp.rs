//! Half precision and automatic mixed precision. Set `RUSTNN_DEVICE=mps`
//! (or `cuda`) to run on a GPU.

use ruai::amp::{autocast, GradScaler};
use ruai::functional as F;
use ruai::nn::{self, Module, Sequential};
use ruai::generation::Sampler;
use ruai::optim::{Adam, Optimizer, SGD};
use ruai::{rand, serialize, DType, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

/// The global RNG is shared by all test threads.
static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn half_values_and_promotion() {
    let x = Tensor::from_slice(&[1.0, 0.1, 65504.0, 1e-8, -2.5], &[5]).to(dev());
    let h = x.to_dtype(DType::F16);
    let b = x.to_dtype(DType::BF16);
    assert_eq!((h.dtype(), b.dtype()), (DType::F16, DType::BF16));
    let hv = h.to_vec();
    assert_eq!(hv[0], 1.0);
    assert!((hv[1] - 0.1).abs() < 1e-4);
    assert_eq!(hv[2], 65504.0); // max f16
    assert_eq!(hv[3], 0.0); // underflows in f16 ...
    assert!(b.to_vec()[3] > 0.0); // ... but not in bf16 (f32 exponent range)
    // promotion
    assert_eq!((&h * 2.0).dtype(), DType::F16, "scalars adopt the tensor dtype");
    assert_eq!((&h + &h).dtype(), DType::F16);
    assert_eq!((&h + &x).dtype(), DType::F32, "mixed floats promote to f32");
    assert_eq!((&h + &b).dtype(), DType::F32);
    assert_eq!(h.gt_scalar(0.0).dtype(), DType::F16, "masks follow the input dtype");
    assert_eq!(h.sum_all().dtype(), DType::F16);
    assert_eq!(h.argmax(0, false).dtype(), DType::I32);
    assert_eq!(h.softmax(0).dtype(), DType::F16, "outside autocast, softmax keeps the dtype");
}

#[test]
fn casts_are_differentiable() {
    let x = Tensor::from_slice(&[0.5, -1.0, 2.0], &[3]).to(dev()).requires_grad_();
    let y = x.to_dtype(DType::F16);
    (&y * &y).sum_all().backward();
    let g = x.grad().unwrap();
    assert_eq!(g.dtype(), DType::F32, "the f32 leaf gets an f32 gradient");
    assert_eq!(g.to_vec(), vec![1.0, -2.0, 4.0]);
}

#[test]
fn autocast_dtype_rules() {
    let _l = lock();
    rand::manual_seed(1);
    let lin = nn::Linear::new(8, 4, dev());
    let ln = nn::LayerNorm::new(4, dev());
    let x = Tensor::randn(&[2, 8], dev());
    autocast(DType::F16, || {
        let y = lin.forward(&x);
        assert_eq!(y.dtype(), DType::F16, "linear runs in half");
        assert_eq!(y.gelu().dtype(), DType::F16);
        assert_eq!(y.softmax(-1).dtype(), DType::F32, "softmax runs in f32");
        assert_eq!(ln.forward(&y).dtype(), DType::F32, "layer norm runs in f32");
        assert_eq!(F::cross_entropy(&y, &[0usize, 3], 0.0).dtype(), DType::F32, "losses run in f32");
    });
    assert_eq!(lin.forward(&x).dtype(), DType::F32, "autocast ends with the closure");
}

/// A small transformer-ish model exercising embedding, attention, layer
/// norm, GELU and linear layers.
fn tiny_gpt() -> nn::GPT {
    nn::GPT::new(nn::GPTConfig { vocab_size: 11, max_len: 8, dim: 16, n_heads: 2, n_layers: 1, dropout: 0.0 }, dev())
}

#[test]
fn mixed_precision_gradients_match_f32() {
    let _l = lock();
    rand::manual_seed(2);
    let model = tiny_gpt();
    let ids: Vec<usize> = (0..2 * 8).map(|i| (i * 5 + 3) % 11).collect();
    let tgt: Vec<usize> = ids.iter().map(|&i| (i + 1) % 11).collect();
    let loss_of = || F::cross_entropy(&model.forward_ids(&ids, 2, 8).reshape(&[16, 11]), &tgt, 0.0);
    loss_of().backward();
    let reference: Vec<Vec<f32>> = model.parameters().iter().map(|p| p.grad().unwrap().to_vec()).collect();
    // Some gradients are exactly zero in theory (e.g. the attention key bias:
    // softmax is shift invariant), so measure errors against at least 1% of
    // the largest gradient anywhere in the model.
    let floor = 0.01 * reference.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    for (d, tol) in [(DType::F16, 0.02), (DType::BF16, 0.08)] {
        model.zero_grad();
        let loss = autocast(d, loss_of);
        assert_eq!(loss.dtype(), DType::F32);
        loss.backward();
        for ((name, p), r) in model.named_parameters().iter().zip(&reference) {
            let g = p.grad().unwrap_or_else(|| panic!("{name}: no grad under autocast"));
            assert_eq!(g.dtype(), DType::F32, "{name}: master gradient must be f32");
            let scale = r.iter().fold(floor, |m, v| m.max(v.abs()));
            let err = g.to_vec().iter().zip(r).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
            assert!(err < tol, "{d} {name}: relative gradient error {err}");
        }
    }
}

#[test]
fn grad_scaler_unscales_and_skips_overflow() {
    let _l = lock();
    rand::manual_seed(3);
    let model = nn::Linear::new(4, 2, dev());
    let x = Tensor::randn(&[3, 4], dev());
    let y = [0usize, 1, 1];
    F::cross_entropy(&model.forward(&x), &y, 0.0).backward();
    let plain: Vec<Vec<f32>> = model.parameters().iter().map(|p| p.grad().unwrap().to_vec()).collect();

    // scaled backward + unscale == plain gradients
    let mut opt = SGD::new(model.parameters(), 0.1);
    let mut scaler = GradScaler::with_scale(1024.0);
    opt.zero_grad();
    scaler.scale(&F::cross_entropy(&model.forward(&x), &y, 0.0)).backward();
    assert!(scaler.unscale(&opt));
    for (p, r) in model.parameters().iter().zip(&plain) {
        assert!(max_rel_err(&p.grad().unwrap().to_vec(), r) < 1e-5);
    }
    let before = model.weight.to_vec();
    assert!(scaler.step(&mut opt), "finite gradients: step taken");
    assert_ne!(model.weight.to_vec(), before);

    // overflowing gradients: step skipped, scale halved
    let before = model.weight.to_vec();
    opt.zero_grad();
    scaler.scale(&(model.forward(&x).sum_all() * f32::INFINITY)).backward();
    assert!(!scaler.step(&mut opt));
    assert_eq!(model.weight.to_vec(), before, "skipped step leaves weights untouched");
    assert_eq!(scaler.scale_value(), 512.0);
    assert_eq!(scaler.skipped_steps(), 1);

    // growth after `growth_interval` clean steps; state round-trips
    scaler.growth_interval = 2;
    for _ in 0..2 {
        opt.zero_grad();
        scaler.scale(&F::cross_entropy(&model.forward(&x), &y, 0.0)).backward();
        assert!(scaler.step(&mut opt));
    }
    assert_eq!(scaler.scale_value(), 1024.0);
    let mut restored = GradScaler::new();
    restored.load_state(&scaler.state());
    assert_eq!((restored.scale_value(), restored.skipped_steps()), (1024.0, 1));
}

fn train_gpt_amp(d: Option<DType>) -> (f32, Vec<usize>) {
    rand::manual_seed(6);
    let model = nn::GPT::new(nn::GPTConfig { vocab_size: 5, max_len: 8, dim: 16, n_heads: 2, n_layers: 1, dropout: 0.0 }, dev());
    let seq = [0usize, 1, 2, 3, 4, 0, 1, 2, 3];
    let (inp, tgt) = (&seq[..8], &seq[1..]);
    let mut opt = Adam::new(model.parameters(), 0.01);
    let mut scaler = GradScaler::new();
    let mut last = 0.0;
    for _ in 0..150 {
        opt.zero_grad();
        let step = || F::cross_entropy(&model.forward_ids(inp, 1, 8).reshape(&[8, 5]), tgt, 0.0);
        let loss = match d {
            Some(d) => autocast(d, step),
            None => step(),
        };
        scaler.scale(&loss).backward();
        scaler.step(&mut opt);
        last = loss.item();
    }
    model.eval();
    let out = match d {
        Some(d) => autocast(d, || model.generate(&[0], 7, &mut Sampler::greedy())),
        None => model.generate(&[0], 7, &mut Sampler::greedy()),
    };
    (last, out)
}

#[test]
fn mixed_precision_training_converges() {
    let _l = lock();
    for d in [None, Some(DType::F16), Some(DType::BF16)] {
        let (loss, out) = train_gpt_amp(d);
        assert!(loss < 0.05, "{d:?}: final loss {loss}");
        assert_eq!(out, vec![0, 1, 2, 3, 4, 0, 1, 2], "{d:?}");
    }
}

#[test]
fn half_precision_inference_and_cnn() {
    let _l = lock();
    rand::manual_seed(4);
    let model = Sequential::new()
        .add(nn::Conv2d::new(2, 4, 3, 1, 1, dev()))
        .add(nn::BatchNorm2d::new(4, dev()))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2, 2))
        .add(nn::Flatten)
        .add(nn::Linear::new(4 * 3 * 3, 5, dev()));
    model.eval();
    let x = Tensor::randn(&[2, 2, 6, 6], dev());
    let reference = model.forward(&x).to_vec();
    // autocast inference
    let y = autocast(DType::F16, || model.forward(&x));
    assert!(max_rel_err(&y.to_vec(), &reference) < 1e-2, "autocast cnn");
    // whole model converted to half
    model.to_dtype(DType::F16);
    assert!(model.parameters().iter().all(|p| p.dtype() == DType::F16));
    let y = model.forward(&x.to_dtype(DType::F16));
    assert_eq!(y.dtype(), DType::F16);
    assert!(max_rel_err(&y.to_vec(), &reference) < 1e-2, "f16 cnn");
}

#[test]
fn safetensors_keeps_half_dtypes() {
    let t = Tensor::from_slice(&[1.5, -0.25, 3.0], &[3]).to(dev());
    let tensors = vec![
        ("h".to_string(), t.to_dtype(DType::F16)),
        ("b".to_string(), t.to_dtype(DType::BF16)),
    ];
    let bytes = serialize::to_safetensors_bytes(&tensors, None).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    assert!(header.contains(r#""h":{"dtype":"F16""#) && header.contains(r#""b":{"dtype":"BF16""#), "{header}");
    let (back, _) = serialize::from_safetensors_bytes(&bytes).unwrap();
    for (name, d) in [("b", DType::BF16), ("h", DType::F16)] {
        let x = &back.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(x.dtype(), d);
        assert_eq!(x.to_vec(), vec![1.5, -0.25, 3.0]);
    }
    // half weights load into an f32 model (converted on copy)
    let lin = nn::Linear::with_bias(3, 1, false, dev());
    let src = vec![("weight".to_string(), Tensor::from_slice(&[1.0, 2.0, 3.0], &[3, 1]).to_dtype(DType::F16))];
    serialize::load_state_dict(&lin, &src, true).unwrap();
    assert_eq!((lin.weight.dtype(), lin.weight.to_vec()), (DType::F32, vec![1.0, 2.0, 3.0]));
}

#[test]
fn narrow_and_cat_in_half_precision() {
    for d in [DType::F16, DType::BF16] {
        let x = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3], dev()).to_dtype(d).requires_grad_();
        let y = x.narrow(1, 1, 1);
        assert_eq!(y.dtype(), d);
        (&y * 2.0).sum_all().backward();
        assert_eq!(x.grad().unwrap().dtype(), d);
        assert_eq!(x.grad().unwrap().to_vec(), vec![0., 2., 0., 0., 2., 0.]);
        // mixed dtypes promote like binary ops
        let a = Tensor::ones(&[1, 2], dev()).to_dtype(d);
        let b = Tensor::zeros(&[1, 1], dev());
        let c = Tensor::cat(&[&a, &b], 1);
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.to_vec(), vec![1., 1., 0.]);
    }
}
