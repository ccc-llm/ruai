//! Finite-difference gradient checks for every differentiable op.

use ruai::functional as F;
use ruai::nn::{self, Module};
use ruai::{no_grad, rand, Device, Tensor};

/// Device under test: set `RUSTNN_DEVICE=mps` to run this file on the GPU.
fn dev() -> Device {
    Device::from_env()
}

/// Compares autograd gradients of `sum(f(inputs) * R)` (R random, fixed)
/// against central finite differences.
fn gradcheck(name: &str, shapes: &[&[usize]], f: impl Fn(&[Tensor]) -> Tensor) {
    rand::manual_seed(42);
    let data: Vec<Vec<f32>> = shapes.iter().map(|s| rand::normal_vec(s.iter().product(), 0.0, 1.0)).collect();
    let make = |d: &[Vec<f32>], grad: bool| -> Vec<Tensor> {
        d.iter()
            .zip(shapes)
            .map(|(v, s)| {
                let t = Tensor::from_vec(v.clone(), s, dev());
                if grad {
                    t.requires_grad_()
                } else {
                    t
                }
            })
            .collect()
    };
    let inputs = make(&data, true);
    let out = f(&inputs);
    let r = Tensor::from_vec(rand::normal_vec(out.numel(), 0.0, 1.0), out.shape(), dev());
    let loss_of = |ts: &[Tensor]| (&f(ts) * &r).sum_all().item() as f64;
    (&out * &r).sum_all().backward();

    let eps = 1e-2f32;
    for (i, inp) in inputs.iter().enumerate() {
        let analytic = inp.grad().map(|g| g.to_vec()).unwrap_or_else(|| vec![0.0; inp.numel()]);
        for j in 0..data[i].len() {
            let mut plus = data.clone();
            plus[i][j] += eps;
            let mut minus = data.clone();
            minus[i][j] -= eps;
            let num = no_grad(|| (loss_of(&make(&plus, false)) - loss_of(&make(&minus, false))) / (2.0 * eps as f64));
            let a = analytic[j] as f64;
            let tol = 2e-2 * (1.0f64).max(a.abs()).max(num.abs());
            assert!((a - num).abs() <= tol, "{name}: input {i} elem {j}: analytic {a:.5} vs numeric {num:.5}");
        }
    }
}

#[test]
fn elementwise() {
    gradcheck("add_broadcast", &[&[3, 4], &[4]], |x| &x[0] + &x[1]);
    gradcheck("sub_broadcast", &[&[2, 3, 1], &[3, 4]], |x| &x[0] - &x[1]);
    gradcheck("mul_broadcast", &[&[2, 3, 4], &[1, 3, 1]], |x| &x[0] * &x[1]);
    gradcheck("div", &[&[3, 4], &[3, 4]], |x| &x[0] / &(x[1].abs() + 1.0));
    gradcheck("exp_log", &[&[5]], |x| (x[0].exp() + 1.0).log());
    gradcheck("tanh_sigmoid", &[&[6]], |x| &x[0].tanh() * &x[0].sigmoid());
    gradcheck("gelu", &[&[7]], |x| x[0].gelu());
    gradcheck("sqrt_powf", &[&[5]], |x| (&x[0] * &x[0] + 1.0).sqrt() + x[0].powf(3.0));
    gradcheck("sin_cos", &[&[5]], |x| &x[0].sin() * &x[0].cos());
    gradcheck("maximum", &[&[8], &[8]], |x| x[0].maximum(&x[1]));
}

#[test]
fn relu_away_from_kink() {
    // keep inputs away from 0 so finite differences are valid
    gradcheck("relu", &[&[10]], |x| (&x[0] * 3.0).relu());
}

#[test]
fn reductions_and_shapes() {
    gradcheck("sum_dim", &[&[3, 4, 5]], |x| x[0].sum(1, false));
    gradcheck("mean_keep", &[&[3, 4]], |x| x[0].mean(-1, true));
    gradcheck("var", &[&[3, 6]], |x| x[0].var(-1, false));
    gradcheck("max", &[&[4, 5]], |x| x[0].max(1, false));
    gradcheck("permute", &[&[2, 3, 4]], |x| &x[0].permute(&[2, 0, 1]) * 2.0);
    gradcheck("transpose_reshape", &[&[2, 6]], |x| x[0].t().reshape(&[3, 4]));
    gradcheck("expand", &[&[3, 1]], |x| x[0].expand(&[2, 3, 4]));
}

#[test]
fn matmul_variants() {
    gradcheck("mm_2d", &[&[3, 4], &[4, 5]], |x| x[0].matmul(&x[1]));
    gradcheck("mm_nd_2d", &[&[2, 3, 4], &[4, 5]], |x| x[0].matmul(&x[1]));
    gradcheck("mm_2d_nd", &[&[3, 4], &[2, 4, 5]], |x| x[0].matmul(&x[1]));
    gradcheck("mm_batched", &[&[2, 2, 3, 4], &[2, 2, 4, 3]], |x| x[0].matmul(&x[1]));
    gradcheck("mm_transposed", &[&[2, 3, 4], &[2, 5, 4]], |x| x[0].matmul(&x[1].t()));
    gradcheck("matmul_nt_2d", &[&[3, 4], &[5, 4]], |x| x[0].matmul_nt(&x[1]));
    gradcheck("matmul_nt_tied_head", &[&[2, 3, 4], &[6, 4]], |x| x[0].matmul_nt(&x[1]));
    gradcheck("matmul_nt_batched", &[&[2, 3, 4], &[2, 5, 4]], |x| x[0].matmul_nt(&x[1]));
}

#[test]
fn nn_functions() {
    gradcheck("softmax", &[&[3, 5]], |x| F::softmax(&x[0], -1));
    gradcheck("log_softmax", &[&[3, 5]], |x| F::log_softmax(&x[0], 1));
    gradcheck("cross_entropy", &[&[4, 3]], |x| F::cross_entropy(&x[0], &[0usize, 2, 1, 2], 0.1));
    gradcheck("cross_entropy_ignore", &[&[4, 3]], |x| F::cross_entropy_ignore(&x[0], &[0usize, 9, 1, 2], 9));
    gradcheck("mse", &[&[4], &[4]], |x| F::mse_loss(&x[0], &x[1]));
    gradcheck("bce_logits", &[&[5]], |x| {
        F::binary_cross_entropy_with_logits(&x[0], &Tensor::from_slice(&[0., 1., 1., 0., 1.], &[5]).to(dev()))
    });
    gradcheck("layer_norm", &[&[3, 6], &[6], &[6]], |x| F::layer_norm(&x[0], Some(&x[1]), Some(&x[2]), 1e-5));
    gradcheck("embedding", &[&[5, 3]], |x| x[0].index_select(&[1usize, 4, 1, 0]));
}

#[test]
fn indexing_and_concat() {
    gradcheck("narrow", &[&[3, 5]], |x| x[0].narrow(1, 1, 3));
    gradcheck("select", &[&[3, 4, 2]], |x| x[0].select(1, 2));
    gradcheck("cat", &[&[2, 3], &[2, 1], &[2, 2]], |x| Tensor::cat(&[&x[0], &x[1], &x[2]], -1));
    gradcheck("stack", &[&[2, 3], &[2, 3]], |x| Tensor::stack(&[&x[0], &x[1]], 0));
    let idx = Tensor::from_indices(&[2, 0, 1, 1, 3, 0], &[3, 2], dev());
    gradcheck("gather", &[&[3, 4]], move |x| x[0].gather(1, &idx));
    let cond = Tensor::from_slice(&[1., 0., 1., 0.], &[4]).to(dev());
    gradcheck("where", &[&[3, 4], &[4]], move |x| Tensor::where_cond(&cond, &x[0], &x[1]));
    gradcheck("minimum", &[&[8], &[8]], |x| x[0].minimum(&x[1]));
    gradcheck("clamp", &[&[10]], |x| (&x[0] * 2.0).clamp(-0.5, 0.7));
}

#[test]
fn convolution_and_pooling() {
    gradcheck("conv2d", &[&[2, 2, 5, 5], &[3, 2, 3, 3], &[3]], |x| F::conv2d(&x[0], &x[1], Some(&x[2]), 1, 1));
    gradcheck("conv2d_stride2", &[&[1, 1, 6, 6], &[2, 1, 2, 2]], |x| F::conv2d(&x[0], &x[1], None, 2, 0));
    gradcheck("max_pool", &[&[1, 2, 4, 4]], |x| F::max_pool2d(&x[0], 2, 2));
    gradcheck("avg_pool", &[&[1, 2, 4, 4]], |x| F::avg_pool2d(&x[0], 2, 2));
    gradcheck("max_pool_padded", &[&[1, 2, 5, 5]], |x| F::max_pool2d_padded(&x[0], 3, 2, 1));
}

#[test]
fn attention() {
    gradcheck("sdpa_causal", &[&[2, 4, 3], &[2, 4, 3], &[2, 4, 3]], |x| {
        let m = F::causal_mask(4, dev());
        F::scaled_dot_product_attention(&x[0], &x[1], &x[2], Some(&m), 0.0, false)
    });
}

#[test]
fn transformer_block_params_get_grads() {
    rand::manual_seed(0);
    let blk = nn::TransformerBlock::new(8, 2, 16, 0.0, true, dev());
    let x = Tensor::randn(&[2, 5, 8], dev()).requires_grad_();
    blk.forward(&x).sum_all().backward();
    assert!(x.grad().is_some());
    for (name, p) in blk.named_parameters() {
        let g = p.grad().unwrap_or_else(|| panic!("{name} has no grad"));
        assert_eq!(g.shape(), p.shape(), "{name}");
    }
}

#[test]
fn causal_attention_does_not_look_ahead() {
    rand::manual_seed(1);
    let blk = nn::TransformerBlock::new(8, 2, 16, 0.0, true, dev());
    blk.eval();
    let x = Tensor::randn(&[1, 6, 8], dev());
    let y1 = blk.forward(&x).to_vec();
    // change only the last time step
    let mut xv = x.to_vec();
    for v in xv[5 * 8..].iter_mut() {
        *v += 1.0;
    }
    let y2 = blk.forward(&Tensor::from_vec(xv, &[1, 6, 8], dev())).to_vec();
    for i in 0..5 * 8 {
        assert!((y1[i] - y2[i]).abs() < 1e-5, "position {} changed", i / 8);
    }
}

#[test]
fn llama_components() {
    gradcheck("silu", &[&[7]], |x| x[0].silu());
    gradcheck("rms_norm", &[&[3, 6], &[6]], |x| F::rms_norm(&x[0], Some(&x[1]), 1e-5));
    let inv = F::rope_inv_freq(4, 10000.0);
    let (cos, sin) = F::rope_cos_sin(&inv, 2, 3, dev());
    gradcheck("rope", &[&[2, 2, 3, 4]], move |x| F::apply_rope(&x[0], &cos, &sin));
    gradcheck("repeat_kv", &[&[1, 2, 3, 2]], |x| F::repeat_kv(&x[0], 3));
    let ffn = nn::SwiGLU::new(4, 8, dev());
    gradcheck("swiglu_input", &[&[2, 4]], move |x| ffn.forward(&x[0]));
}
