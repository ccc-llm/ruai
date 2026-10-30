//! safetensors interop, legacy format, state-dict loading, and checkpoints
//! that resume training exactly.

use ruai::checkpoint::{self, TrainState};
use ruai::functional as F;
use ruai::nn::{self, Module, Sequential};
use ruai::optim::{Adam, Optimizer, SGD};
use ruai::serialize::{self, Metadata};
use ruai::{rand, DType, Device, Tensor};

/// Device under test: set `RUSTNN_DEVICE=mps` to run this file on the GPU.
fn dev() -> Device {
    Device::from_env()
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ruai_{}_{name}", std::process::id()))
}

/// Build a safetensors file by hand, straight from the format spec:
/// u64 LE header length, JSON header, raw little-endian data.
fn handmade_safetensors(entries: &[(&str, &str, &[usize], Vec<u8>)], metadata: &str) -> Vec<u8> {
    let mut header = String::from("{");
    let mut data = Vec::new();
    if !metadata.is_empty() {
        header.push_str(&format!("\"__metadata__\":{metadata},"));
    }
    for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            shape.join(","),
            data.len(),
            data.len() + bytes.len()
        ));
        if i + 1 < entries.len() {
            header.push(',');
        }
        data.extend_from_slice(bytes);
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[test]
fn reads_spec_files_with_many_dtypes() {
    let f16 = [0x3c00u16, 0xc000, 0x3555, 0x0001, 0x7c00]; // 1, -2, ~0.333, smallest subnormal, inf
    let bf16 = [0x3f80u16, 0xc040, 0x4049]; // 1, -3, 3.140625
    let f64v = [2.5f64, -0.125];
    let i64v = [7i64, -9, 1 << 30];
    let u8v = [0u8, 200, 255];
    let le16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let bytes = handmade_safetensors(
        &[
            ("a.f16", "F16", &[5], le16(&f16)),
            ("b.bf16", "BF16", &[3, 1], le16(&bf16)),
            ("c.f64", "F64", &[2], f64v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ("d.i64", "I64", &[3], i64v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ("e.u8", "U8", &[3], u8v.to_vec()),
            ("f.scalar", "F32", &[], 4.5f32.to_le_bytes().to_vec()),
        ],
        r#"{"source":"handmade","step":"12"}"#,
    );
    let (tensors, meta) = serialize::from_safetensors_bytes(&bytes).unwrap();
    let get = |n: &str| tensors.iter().find(|(k, _)| k == n).unwrap().1.clone();
    let a = get("a.f16").to_vec();
    assert_eq!(&a[..2], &[1.0, -2.0]);
    assert!((a[2] - 0.33325).abs() < 1e-4);
    assert_eq!(a[3], 2f32.powi(-24));
    assert!(a[4].is_infinite());
    let b = get("b.bf16");
    assert_eq!((b.shape(), b.to_vec()), (&[3usize, 1][..], vec![1.0, -3.0, 3.140625]));
    assert_eq!(get("c.f64").to_vec(), vec![2.5, -0.125]);
    let d = get("d.i64");
    assert_eq!((d.dtype(), d.to_vec_i32()), (DType::I32, vec![7, -9, 1 << 30]));
    assert_eq!(get("e.u8").to_vec_i32(), vec![0, 200, 255]);
    assert_eq!(get("f.scalar").shape(), &[] as &[usize]);
    assert_eq!(meta.get("source").map(String::as_str), Some("handmade"));
    // names come back sorted
    let names: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["a.f16", "b.bf16", "c.f64", "d.i64", "e.u8", "f.scalar"]);
}

#[test]
fn writes_spec_compliant_files() {
    let t = vec![
        ("w".to_string(), Tensor::from_slice(&[1.5, -2.0, 3.25, 0.0, 1.0, 2.0], &[2, 3]).to(dev())),
        ("ids".to_string(), Tensor::from_indices(&[4, 2], &[2], dev())),
    ];
    let mut meta = Metadata::new();
    meta.insert("note".into(), "hi".into());
    let bytes = serialize::to_safetensors_bytes(&t, Some(meta)).unwrap();
    // parse the header ourselves
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    assert!(header.contains(r#""w":{"dtype":"F32","shape":[2,3],"data_offsets":["#), "{header}");
    assert!(header.contains(r#""ids":{"dtype":"I32","shape":[2],"#), "{header}");
    assert!(header.contains(r#""__metadata__":{"note":"hi"}"#), "{header}");
    // and the data region holds the raw little-endian floats of "w"
    let body = &bytes[8 + n..];
    let floats: Vec<f32> = body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert!(floats.windows(6).any(|w| w == [1.5, -2.0, 3.25, 0.0, 1.0, 2.0]));
    // round trip
    let (back, meta) = serialize::from_safetensors_bytes(&bytes).unwrap();
    assert_eq!(back[1].0, "w");
    assert_eq!(back[1].1.to_vec(), t[0].1.to_vec());
    assert_eq!(back[0].1.to_vec_i32(), vec![4, 2]);
    assert_eq!(meta["note"], "hi");
    // duplicate names are rejected
    assert!(serialize::to_safetensors_bytes(&[t[0].clone(), t[0].clone()], None).is_err());
}

#[test]
fn loads_legacy_rsnn_files() {
    let mut b = b"RSNN".to_vec();
    b.extend(1u32.to_le_bytes()); // version
    b.extend(1u32.to_le_bytes()); // count
    b.extend(6u32.to_le_bytes());
    b.extend(b"weight");
    b.extend(2u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    b.extend(0.5f32.to_le_bytes());
    b.extend((-1.0f32).to_le_bytes());
    let p = tmp("legacy.rsnn");
    std::fs::write(&p, b).unwrap();
    let t = serialize::load_tensors(&p).unwrap();
    std::fs::remove_file(&p).ok();
    assert_eq!(t[0].0, "weight");
    assert_eq!((t[0].1.shape(), t[0].1.to_vec()), (&[1usize, 2][..], vec![0.5, -1.0]));
}

#[test]
fn load_state_dict_reports_mismatches() {
    let _rng = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // Linear::new draws from the RNG
    let model = nn::Linear::new(3, 2, dev());
    let w = Tensor::ones(&[3, 2], Device::Cpu);
    let extra = Tensor::zeros(&[1], Device::Cpu);
    let src = vec![("weight".to_string(), w), ("junk".to_string(), extra)];
    assert!(serialize::load_state_dict(&model, &src, true).is_err());
    let r = serialize::load_state_dict(&model, &src, false).unwrap();
    assert_eq!(r.missing, vec!["bias".to_string()]);
    assert_eq!(r.unexpected, vec!["junk".to_string()]);
    assert_eq!(model.weight.to_vec(), vec![1.0; 6]);
    assert_eq!(model.weight.device(), dev());
    let bad = vec![("weight".to_string(), Tensor::ones(&[2, 3], Device::Cpu))];
    let err = serialize::load_state_dict(&model, &bad, false).unwrap_err();
    assert!(err.to_string().contains("shape mismatch"), "{err}");
}

/// The global RNG is process-wide and tests run on parallel threads:
/// RNG-dependent tests take this lock so they do not perturb each other.
static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn make_model(seed: u64) -> Sequential {
    rand::manual_seed(seed);
    Sequential::new()
        .add(nn::Linear::new(4, 16, dev()))
        .add(nn::GELU)
        .add(nn::Dropout::new(0.3)) // consumes the global RNG
        .add(nn::Linear::new(16, 3, dev()))
}

/// Train 10 steps straight vs. 5 steps + checkpoint + fresh process-like
/// restore + 5 steps: the final weights must be bit-identical.
fn resume_is_exact(make_opt: impl Fn(&Sequential) -> Box<dyn Optimizer>, tag: &str) {
    let _rng = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // reference run
    let a = make_model(1);
    let mut oa = make_opt(&a);
    rand::manual_seed(99);
    train_steps_rng(&a, oa.as_mut(), 0..10);

    // interrupted run
    let b = make_model(1);
    let mut ob = make_opt(&b);
    rand::manual_seed(99);
    train_steps_rng(&b, ob.as_mut(), 0..5);
    let p = tmp(&format!("ckpt_{tag}.safetensors"));
    let mut extra = Metadata::new();
    extra.insert("best_loss".into(), "0.5".into());
    checkpoint::save(&p, &b, ob.as_ref(), &TrainState { step: 5, epoch: 1, extra }).unwrap();
    rand::manual_seed(12345); // clobber the RNG: load must restore it

    // "new process": different init, fresh optimizer
    let c = make_model(777);
    let mut oc = make_opt(&c);
    let state = checkpoint::load(&p, &c, oc.as_mut()).unwrap();
    assert_eq!(checkpoint::peek(&p).unwrap().step, 5);
    std::fs::remove_file(&p).ok();
    assert_eq!((state.step, state.epoch), (5, 1));
    assert_eq!(state.extra.get("best_loss").map(String::as_str), Some("0.5"));
    train_steps_rng(&c, oc.as_mut(), 5..10);

    for ((n, x), (_, y)) in a.named_parameters().iter().zip(c.named_parameters().iter()) {
        assert_eq!(x.to_vec(), y.to_vec(), "{tag}: parameter {n} differs after resume");
    }
}

/// Like `train_steps`, but leaves the global RNG alone so dropout masks
/// depend on it (that is what the checkpoint has to restore).
fn train_steps_rng(model: &Sequential, opt: &mut dyn Optimizer, steps: std::ops::Range<u64>) {
    for step in steps {
        let x: Vec<f32> = (0..8 * 4).map(|i| ((i as u64 * 31 + step * 7) % 13) as f32 / 13.0 - 0.5).collect();
        let y: Vec<usize> = (0..8).map(|i| ((i as u64 + step) % 3) as usize).collect();
        opt.zero_grad();
        F::cross_entropy(&model.forward(&Tensor::from_vec(x, &[8, 4], dev())), &y, 0.0).backward();
        opt.step();
    }
}

#[test]
fn checkpoint_resume_is_bit_exact_adamw() {
    resume_is_exact(|m| Box::new(Adam::adamw(m.parameters(), 1e-2, 0.01)), "adamw");
}

#[test]
fn checkpoint_resume_is_bit_exact_sgd_momentum() {
    resume_is_exact(|m| Box::new(SGD::with_momentum(m.parameters(), 0.05, 0.9, 1e-4)), "sgd");
}

#[test]
fn checkpoint_rejects_wrong_optimizer() {
    let _rng = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = make_model(1);
    let mut adam = Adam::new(m.parameters(), 1e-3);
    train_steps_rng(&m, &mut adam, 0..1);
    let p = tmp("ckpt_wrong.safetensors");
    checkpoint::save(&p, &m, &adam, &TrainState::default()).unwrap();
    let mut sgd = SGD::new(m.parameters(), 0.1);
    let err = checkpoint::load(&p, &m, &mut sgd).unwrap_err();
    std::fs::remove_file(&p).ok();
    assert!(err.to_string().contains("optimizer"), "{err}");
}

#[test]
fn gradient_accumulation_matches_full_batch() {
    let _rng = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    rand::manual_seed(3);
    let model = nn::Linear::new(5, 4, dev());
    let x = Tensor::randn(&[8, 5], dev());
    let y = [0usize, 1, 2, 3, 3, 2, 1, 0];
    F::cross_entropy(&model.forward(&x), &y, 0.0).backward();
    let full: Vec<Vec<f32>> = model.parameters().iter().map(|p| p.grad().unwrap().to_vec()).collect();
    model.zero_grad();
    // two micro-batches of 4, each loss scaled by 1/2: gradients accumulate in .grad
    for k in 0..2 {
        let xs = x.narrow(0, 4 * k, 4);
        (F::cross_entropy(&model.forward(&xs), &y[4 * k..4 * k + 4], 0.0) * 0.5).backward();
    }
    for (p, f) in model.parameters().iter().zip(&full) {
        for (a, b) in p.grad().unwrap().to_vec().iter().zip(f) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
