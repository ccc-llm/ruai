//! Backend-level checks: fused kernels agree with the composed reference
//! implementations, dtypes behave, new ops give the right values.

use ruai::backend::{backend_for, reference::Unfused, AdamParams, Backend};
use ruai::functional as F;
use ruai::{rand, DType, Device, Tensor};

const CPU: Device = Device::Cpu;

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "elem {i}: {x} vs {y}");
    }
}

#[test]
fn fused_ops_match_reference() {
    rand::manual_seed(7);
    let fast = backend_for(CPU);
    let slow = Unfused(fast);
    for shape in [vec![5, 7], vec![2, 3, 4], vec![4, 1]] {
        let n: usize = shape.iter().product();
        let x = fast.from_f32(rand::normal_vec(n, 0.0, 2.0));
        let g = fast.from_f32(rand::normal_vec(n, 0.0, 1.0));
        for axis in 0..shape.len() {
            let (a, b) = (fast.softmax(&x, &shape, axis), slow.softmax(&x, &shape, axis));
            close(&a.to_vec(), &b.to_vec(), 1e-5);
            close(
                &fast.softmax_backward(&a, &g, &shape, axis).to_vec(),
                &slow.softmax_backward(&b, &g, &shape, axis).to_vec(),
                1e-5,
            );
            let (a, b) = (fast.log_softmax(&x, &shape, axis), slow.log_softmax(&x, &shape, axis));
            close(&a.to_vec(), &b.to_vec(), 1e-5);
            close(
                &fast.log_softmax_backward(&a, &g, &shape, axis).to_vec(),
                &slow.log_softmax_backward(&b, &g, &shape, axis).to_vec(),
                1e-5,
            );
        }
        let (rows, dim) = (n / shape[shape.len() - 1], shape[shape.len() - 1]);
        let (xa, ra) = fast.layer_norm(&x, rows, dim, 1e-5);
        let (xb, rb) = slow.layer_norm(&x, rows, dim, 1e-5);
        close(&xa.to_vec(), &xb.to_vec(), 1e-4);
        close(&ra.to_vec(), &rb.to_vec(), 1e-4);
        close(
            &fast.layer_norm_backward(&g, &xa, &ra, rows, dim).to_vec(),
            &slow.layer_norm_backward(&g, &xb, &rb, rows, dim).to_vec(),
            1e-4,
        );
    }
}

#[test]
fn matmul_matches_naive() {
    rand::manual_seed(8);
    // sizes large enough to take the parallel / row-split path
    for &(b, m, k, n) in &[(1, 3, 4, 5), (1, 130, 70, 90), (3, 65, 33, 129), (16, 8, 64, 8)] {
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = Tensor::randn(&if ta { [b, k, m] } else { [b, m, k] }, CPU);
            let bt = Tensor::randn(&if tb { [b, n, k] } else { [b, k, n] }, CPU);
            let a2 = if ta { a.t() } else { a.clone() };
            let b2 = if tb { bt.t() } else { bt.clone() };
            let got = a2.matmul(&b2).to_vec();
            let (av, bv) = (a2.to_vec(), b2.to_vec());
            let mut want = vec![0.0f32; b * m * n];
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        want[(bi * m + i) * n + j] =
                            (0..k).map(|p| av[(bi * m + i) * k + p] * bv[(bi * k + p) * n + j]).sum();
                    }
                }
            }
            close(&got, &want, 1e-4);
        }
    }
}

#[test]
fn dtypes() {
    let ids = Tensor::from_indices(&[3, 1, 2], &[3], CPU);
    assert_eq!(ids.dtype(), DType::I32);
    assert_eq!(ids.to_vec_i32(), vec![3, 1, 2]);
    assert_eq!(ids.to_dtype(DType::F32).to_vec(), vec![3.0, 1.0, 2.0]);
    let x = Tensor::from_slice(&[0.1, 0.9, 0.3, 0.8, 0.2, 0.1], &[2, 3]);
    let am = x.argmax(-1, false);
    assert_eq!(am.dtype(), DType::I32);
    assert_eq!(am.to_vec_i32(), vec![1, 0]);
    assert_eq!(F::accuracy(&x, &am), 1.0);
    assert_eq!(F::accuracy(&x, &[1usize, 2]), 0.5);
    // reshape / narrow / cat keep the dtype
    let c = Tensor::cat(&[&ids, &ids.narrow(0, 1, 2)], 0);
    assert_eq!(c.dtype(), DType::I32);
    assert_eq!(c.to_vec_i32(), vec![3, 1, 2, 1, 2]);
}

#[test]
#[should_panic(expected = "only floating-point tensors can require grad")]
fn int_tensors_cannot_require_grad() {
    Tensor::from_indices(&[1], &[1], CPU).requires_grad_();
}

#[test]
fn new_ops_values() {
    let x = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_eq!(x.narrow(1, 1, 2).to_vec(), vec![2., 3., 5., 6.]);
    assert_eq!(x.select(0, 1).to_vec(), vec![4., 5., 6.]);
    assert_eq!(Tensor::stack(&[&x, &x], 1).shape(), &[2, 2, 3]);
    let idx = Tensor::from_indices(&[2, 0], &[2, 1], CPU);
    assert_eq!(x.gather(1, &idx).to_vec(), vec![3., 4.]);
    assert_eq!(x.clamp(2.0, 4.0).to_vec(), vec![2., 2., 3., 4., 4., 4.]);
    let m = x.gt_scalar(3.5);
    assert_eq!(x.masked_fill(&m, 0.0).to_vec(), vec![1., 2., 3., 0., 0., 0.]);
}

#[test]
fn cross_entropy_matches_onehot_formula() {
    rand::manual_seed(9);
    let logits = Tensor::randn(&[6, 5], CPU);
    let t = [0usize, 4, 2, 2, 1, 3];
    for ls in [0.0, 0.1] {
        let got = F::cross_entropy(&logits, &t, ls).item();
        let lsm = logits.log_softmax(-1).to_vec();
        let mut want = 0.0;
        for (i, &ti) in t.iter().enumerate() {
            for c in 0..5 {
                let w = if c == ti { 1.0 - ls + ls / 5.0 } else { ls / 5.0 };
                want -= w * lsm[i * 5 + c];
            }
        }
        want /= 6.0;
        assert!((got - want).abs() < 1e-5, "ls={ls}: {got} vs {want}");
    }
}

#[test]
fn fused_adam_matches_reference() {
    rand::manual_seed(10);
    let fast = backend_for(CPU);
    let slow = Unfused(fast);
    let n = 40000; // spans several parallel chunks
    let mk = |s: f32| fast.from_f32(rand::normal_vec(n, 0.0, s));
    let (p, g, m) = (mk(1.0), mk(0.1), mk(0.01));
    let v = fast.from_f32(rand::uniform_vec(n, 0.0, 0.01));
    for decoupled in [false, true] {
        let h = AdamParams {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
            decoupled,
            bias_correction1: 1.0 - 0.9f32.powi(3),
            bias_correction2: 1.0 - 0.999f32.powi(3),
        };
        let a = fast.adam_step(&p, &g, &m, &v, h);
        let b = slow.adam_step(&p, &g, &m, &v, h);
        close(&a.0.to_vec(), &b.0.to_vec(), 1e-5);
        close(&a.1.to_vec(), &b.1.to_vec(), 1e-5);
        close(&a.2.to_vec(), &b.2.to_vec(), 1e-5);
    }
}
