//! Fused attention (`Backend::attention`, online softmax) against the
//! composed reference. Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::functional as F;
use ruai::{rand, DType, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn randn(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, dev())
}

/// Masks of every supported broadcast shape for [b, h, tq, tk].
fn masks(b: usize, h: usize, tq: usize, tk: usize) -> Vec<(&'static str, Option<Tensor>)> {
    let pad: Vec<f32> = (0..b * tk).map(|i| if i % tk >= tk - 1 - (i / tk) % 2 { -1e9 } else { 0.0 }).collect();
    vec![
        ("none", None),
        ("causal", Some(F::causal_mask_offset(tq, tk - tq, dev()))),
        ("padding [b,1,1,tk]", Some(Tensor::from_vec(pad, &[b, 1, 1, tk], dev()))),
        ("per head [1,h,tq,tk]", Some(randn(&[1, h, tq, tk]))),
        ("full [b,h,tq,tk]", Some(randn(&[b, h, tq, tk]))),
        ("row [tk]", Some(randn(&[tk]))),
    ]
}

#[test]
fn fused_matches_composed() {
    rand::manual_seed(0);
    for &(b, h, tq, tk, d, dv) in &[(2, 3, 5, 5, 8, 8), (1, 2, 3, 7, 16, 4), (2, 1, 1, 9, 64, 64), (1, 4, 6, 6, 40, 40)] {
        let (q, k, v) = (randn(&[b, h, tq, d]), randn(&[b, h, tk, d]), randn(&[b, h, tk, dv]));
        for (name, m) in masks(b, h, tq, tk) {
            let fused = F::scaled_dot_product_attention(&q, &k, &v, m.as_ref(), 0.0, false);
            assert_eq!(fused.shape(), &[b, h, tq, dv]);
            let want = F::attention_composed(&q, &k, &v, m.as_ref(), 0.0, false).to_vec();
            let err = max_abs_diff(&fused.to_vec(), &want);
            assert!(err < 1e-4, "{name} b={b} h={h} tq={tq} tk={tk} d={d} dv={dv}: {err}");
        }
    }
}

#[test]
fn fused_takes_the_kernel_path() {
    let (q, k, v) = (randn(&[1, 2, 3, 4]).requires_grad_(), randn(&[1, 2, 3, 4]), randn(&[1, 2, 3, 4]));
    let y = F::scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
    assert_eq!(y.op_name(), Some("attention"));
    // active dropout and non-rank-4 inputs use the composed path
    let y = F::scaled_dot_product_attention(&q, &k, &v, None, 0.5, true);
    assert_ne!(y.op_name(), Some("attention"));
    let q3 = randn(&[2, 3, 4]).requires_grad_();
    assert_ne!(F::scaled_dot_product_attention(&q3, &q3, &q3, None, 0.0, false).op_name(), Some("attention"));
}

#[test]
fn fused_gradients_match_composed() {
    rand::manual_seed(1);
    let (b, h, tq, tk, d) = (2, 2, 4, 6, 8);
    let data: Vec<Tensor> = vec![randn(&[b, h, tq, d]), randn(&[b, h, tk, d]), randn(&[b, h, tk, d])];
    let r = randn(&[b, h, tq, d]);
    for (name, m) in masks(b, h, tq, tk) {
        let grads = |fused: bool| {
            let xs: Vec<Tensor> = data.iter().map(|t| t.detach().requires_grad_()).collect();
            let y = if fused {
                F::scaled_dot_product_attention(&xs[0], &xs[1], &xs[2], m.as_ref(), 0.0, false)
            } else {
                F::attention_composed(&xs[0], &xs[1], &xs[2], m.as_ref(), 0.0, false)
            };
            (&y * &r).sum_all().backward();
            xs.iter().map(|x| x.grad().unwrap().to_vec()).collect::<Vec<_>>()
        };
        for (i, (a, c)) in grads(true).iter().zip(grads(false)).enumerate() {
            assert!(max_abs_diff(a, &c) < 1e-4, "{name}: input {i} grads differ by {}", max_abs_diff(a, &c));
        }
    }
}

#[test]
fn half_precision() {
    rand::manual_seed(2);
    for dt in [DType::F16, DType::BF16] {
        let (q, k, v) = (randn(&[2, 2, 5, 32]), randn(&[2, 2, 7, 32]), randn(&[2, 2, 7, 32]));
        let m = F::causal_mask_offset(5, 2, dev());
        let want = F::attention_composed(&q, &k, &v, Some(&m), 0.0, false).to_vec();
        let y = F::scaled_dot_product_attention(&q.to_dtype(dt), &k.to_dtype(dt), &v.to_dtype(dt), Some(&m), 0.0, false);
        assert_eq!(y.dtype(), dt);
        let tol = if dt == DType::F16 { 1e-2 } else { 5e-2 };
        assert!(max_abs_diff(&y.to_vec(), &want) < tol, "{dt}: {}", max_abs_diff(&y.to_vec(), &want));
        // under autocast f32 inputs run in half precision, like the composed path
        let y = ruai::autocast(dt, || F::scaled_dot_product_attention(&q, &k, &v, Some(&m), 0.0, false));
        assert_eq!(y.dtype(), dt);
    }
}

#[test]
fn long_sequence_is_stable() {
    // large logits: the online softmax must rescale correctly
    rand::manual_seed(3);
    let (q, k, v) = (randn(&[1, 1, 3, 16]) * 8.0, randn(&[1, 1, 300, 16]) * 8.0, randn(&[1, 1, 300, 16]));
    let fused = F::scaled_dot_product_attention(&q, &k, &v, None, 0.0, false).to_vec();
    let want = F::attention_composed(&q, &k, &v, None, 0.0, false).to_vec();
    assert!(fused.iter().all(|x| x.is_finite()));
    assert!(max_abs_diff(&fused, &want) < 1e-4);
}

#[test]
fn chunked_default_matches_for_any_block_size() {
    use ruai::backend::{attention_chunked, backend_for, AttentionSpec};
    rand::manual_seed(4);
    let (b, h, tq, tk, d) = (2, 2, 7, 9, 8);
    let (q, k, v) = (randn(&[b, h, tq, d]), randn(&[b, h, tk, d]), randn(&[b, h, tk, d]));
    let be = backend_for(dev());
    for (name, m) in masks(b, h, tq, tk) {
        let want = F::attention_composed(&q, &k, &v, m.as_ref(), 0.0, false).to_vec();
        let mut mask_shape = None;
        if let Some(m) = &m {
            let mut ms = [1usize; 4];
            ms[4 - m.rank()..].copy_from_slice(m.shape());
            mask_shape = Some(ms);
        }
        let spec = AttentionSpec { b, h, tq, tk, d, dv: d, scale: 1.0 / (d as f32).sqrt(), mask_shape };
        // budgets giving blocks of 1, 2 (uneven: 7 = 2+2+2+1), 3 and all 7 query rows
        for rows in [1, 2, 3, 7] {
            let m_storage = m.as_ref().map(|m| m.storage());
            let got = attention_chunked(be, &q.storage(), &k.storage(), &v.storage(), m_storage.as_ref(), spec, rows * b * h * tk);
            assert!(max_abs_diff(&got.to_vec(), &want) < 1e-4, "{name}, {rows} rows per block");
        }
    }
}
