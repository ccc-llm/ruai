//! Second-order gradients: `autograd::grad(.., create_graph = true)` and
//! `backward_create_graph` give differentiable gradients. Each op's
//! Hessian-vector product is checked against finite differences of its
//! first-order gradient. Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::functional as F;
use ruai::nn::{self, Module};
use ruai::{autograd, rand, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// First-order gradient of `sum(f(x) * r)` w.r.t. every input (optionally differentiable).
fn grads(f: &dyn Fn(&[Tensor]) -> Tensor, xs: &[Tensor], r: &Tensor, create: bool) -> Vec<Tensor> {
    let y = (&f(xs) * r).sum_all();
    let refs: Vec<&Tensor> = xs.iter().collect();
    autograd::grad(&y, &refs, create).into_iter().map(|g| g.unwrap_or_else(|| Tensor::zeros_like(&xs[0]))).collect()
}

/// Checks the Hessian-vector product H·u of `sum(f(x) * r)` (autograd of
/// autograd) against central differences of the gradient along `u`.
fn hvp_check(name: &str, shapes: &[&[usize]], f: impl Fn(&[Tensor]) -> Tensor) {
    let _l = lock();
    rand::manual_seed(7);
    let data: Vec<Vec<f32>> = shapes.iter().map(|s| rand::normal_vec(s.iter().product(), 0.0, 1.0)).collect();
    let dirs: Vec<Vec<f32>> = shapes.iter().map(|s| rand::normal_vec(s.iter().product(), 0.0, 1.0)).collect();
    let leaves = |shift: f32| -> Vec<Tensor> {
        data.iter()
            .zip(&dirs)
            .zip(shapes)
            .map(|((d, u), s)| Tensor::from_vec(d.iter().zip(u).map(|(a, b)| a + shift * b).collect(), s, dev()).requires_grad_())
            .collect()
    };
    let xs = leaves(0.0);
    let out_shape = f(&xs).shape().to_vec();
    let r = Tensor::from_vec(rand::normal_vec(out_shape.iter().product(), 0.0, 1.0), &out_shape, dev());
    let us: Vec<Tensor> = dirs.iter().zip(shapes).map(|(u, s)| Tensor::from_vec(u.clone(), s, dev())).collect();

    // analytic: d/dx Σ_i <grad_i(x), u_i>
    let g = grads(&f, &xs, &r, true);
    let dot = g.iter().zip(&us).map(|(gi, ui)| (gi * ui).sum_all()).reduce(|a, b| a + b).unwrap();
    assert!(dot.requires_grad(), "{name}: the gradient has no graph");
    let refs: Vec<&Tensor> = xs.iter().collect();
    let hvp = autograd::grad(&dot, &refs, false);

    // numeric: (grad(x + e u) - grad(x - e u)) / 2e
    let e = 1e-2;
    let (gp, gm) = (grads(&f, &leaves(e), &r, false), grads(&f, &leaves(-e), &r, false));
    for i in 0..xs.len() {
        let num: Vec<f32> = gp[i].to_vec().iter().zip(gm[i].to_vec()).map(|(a, b)| (a - b) / (2.0 * e)).collect();
        let ana = hvp[i].as_ref().map_or_else(|| vec![0.0; num.len()], |h| h.to_vec());
        for (j, (a, n)) in ana.iter().zip(&num).enumerate() {
            let tol = 3e-2 * 1f32.max(a.abs()).max(n.abs());
            assert!((a - n).abs() <= tol, "{name}: input {i} elem {j}: analytic {a:.5} vs numeric {n:.5}");
        }
    }
}

#[test]
fn elementwise() {
    hvp_check("exp", &[&[6]], |x| (&x[0] * 0.5).exp());
    hvp_check("log", &[&[6]], |x| (&x[0] * &x[0] + 1.0).log());
    hvp_check("sqrt_recip", &[&[6]], |x| (&x[0] * &x[0] + 1.0).sqrt() + (&x[0] * &x[0] + 2.0).recip());
    hvp_check("tanh_sigmoid", &[&[6]], |x| &x[0].tanh() * &x[0].sigmoid());
    hvp_check("gelu_silu", &[&[6]], |x| x[0].gelu() + x[0].silu());
    hvp_check("sin_cos", &[&[6]], |x| &x[0].sin() * &x[0].cos());
    hvp_check("powf", &[&[6]], |x| (&x[0] * &x[0] + 1.0).powf(1.5));
    hvp_check("mul_div", &[&[2, 3], &[3]], |x| &(&x[0] * &x[1]) / &(&x[1] * &x[1] + 1.0));
    hvp_check("where", &[&[6], &[6]], |x| {
        let c = Tensor::from_vec(vec![1., 0., 1., 1., 0., 0.], &[6], dev());
        Tensor::where_cond(&c, &(&x[0] * &x[1]), &x[0].exp())
    });
}

#[test]
fn matmul_variants() {
    hvp_check("mm_2d", &[&[3, 4], &[4, 2]], |x| x[0].matmul(&x[1]).tanh());
    hvp_check("mm_nd_2d", &[&[2, 3, 4], &[4, 2]], |x| x[0].matmul(&x[1]).tanh());
    hvp_check("mm_batched", &[&[2, 3, 4], &[2, 4, 2]], |x| x[0].matmul(&x[1]).tanh());
    hvp_check("matmul_nt", &[&[2, 3, 4], &[5, 4]], |x| x[0].matmul_nt(&x[1]).tanh());
}

#[test]
fn nn_functions() {
    hvp_check("softmax", &[&[3, 5]], |x| F::softmax(&x[0], -1));
    hvp_check("log_softmax", &[&[3, 5]], |x| F::log_softmax(&x[0], 1));
    hvp_check("cross_entropy", &[&[4, 3]], |x| F::cross_entropy(&x[0], &[0usize, 2, 1, 2], 0.0));
    hvp_check("layer_norm", &[&[3, 6], &[6]], |x| F::layer_norm(&x[0], Some(&x[1]), None, 1e-5));
    hvp_check("rms_norm", &[&[3, 6]], |x| F::rms_norm(&x[0], None, 1e-5));
    hvp_check("embedding_gather", &[&[5, 3]], |x| {
        let idx = Tensor::from_indices(&[2, 0, 1], &[3, 1], dev());
        x[0].index_select(&[1usize, 4, 1]).tanh().gather(1, &idx)
    });
    hvp_check("conv2d", &[&[1, 2, 5, 5], &[3, 2, 3, 3]], |x| F::conv2d(&x[0], &x[1], None, 1, 1).tanh());
    hvp_check("attention_fused", &[&[1, 2, 3, 4], &[1, 2, 5, 4], &[1, 2, 5, 4]], |x| {
        let m = F::causal_mask_offset(3, 2, dev());
        F::scaled_dot_product_attention(&x[0], &x[1], &x[2], Some(&m), 0.0, false)
    });
}

#[test]
fn checkpointed_block() {
    let _l = lock();
    rand::manual_seed(3);
    let lin = nn::Linear::new(4, 4, dev());
    let w = lin.weight.clone();
    drop(_l);
    hvp_check("grad_checkpoint", &[&[2, 4]], move |x| {
        let wc = w.clone();
        F::grad_checkpoint(&[&x[0]], vec![w.clone()], move |xs| xs[0].matmul(&wc).tanh())
    });
}

#[test]
fn backward_create_graph_and_grad_semantics() {
    let x = Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3], dev()).requires_grad_();
    (&x * &x * &x).sum_all().backward_create_graph();
    let gx = x.grad().unwrap(); // 3x²
    assert!(gx.requires_grad());
    assert_eq!(gx.to_vec(), vec![3.0, 12.0, 0.75]);
    x.zero_grad();
    gx.sum_all().backward(); // d/dx Σ 3x² = 6x
    assert_eq!(x.grad().unwrap().to_vec(), vec![6.0, -12.0, 3.0]);

    // autograd::grad leaves .grad alone and reports unused inputs as None
    let y = Tensor::from_vec(vec![1.0, 2.0], &[2], dev()).requires_grad_();
    let unused = Tensor::from_vec(vec![1.0], &[1], dev()).requires_grad_();
    let g = autograd::grad(&(&y * 3.0).sum_all(), &[&y, &unused], false);
    assert_eq!(g[0].as_ref().unwrap().to_vec(), vec![3.0, 3.0]);
    assert!(g[1].is_none());
    assert!(y.grad().is_none() && unused.grad().is_none());
    assert!(!g[0].as_ref().unwrap().requires_grad(), "no graph without create_graph");
}

#[test]
fn gradient_penalty_trains_parameters() {
    // WGAN-GP style: penalty = (‖∇_x D(x)‖ - 1)², differentiated w.r.t. D's parameters
    let _l = lock();
    rand::manual_seed(11);
    let d = nn::Sequential::new().add(nn::Linear::new(3, 8, dev())).add(nn::Tanh).add(nn::Linear::new(8, 1, dev()));
    // large inputs: tanh is curved there, so its second derivative matters
    let x = Tensor::randn(&[4, 3], dev()) * 3.0;
    let penalty = || {
        let xg = x.detach().requires_grad_();
        let gx = autograd::grad(&d.forward(&xg).sum_all(), &[&xg], true)[0].clone().unwrap();
        let norm = (&gx * &gx).sum(1, false).add_scalar(1e-12).sqrt();
        let dev1 = norm.add_scalar(-1.0);
        (&dev1 * &dev1).mean_all()
    };
    d.zero_grad();
    penalty().backward();
    let p = d.parameters()[0].clone(); // first layer weight [3, 8]
    let analytic = p.grad().expect("penalty reaches the weights").to_vec();
    // finite differences on a few weight entries
    let base = p.to_vec();
    for &j in &[0usize, 5, 13, 22] {
        let at = |delta: f32| {
            let mut w = base.clone();
            w[j] += delta;
            p.set_storage(Tensor::from_vec(w, &[3, 8], dev()).storage());
            penalty().item()
        };
        let num = (at(1e-2) - at(-1e-2)) / 2e-2;
        p.set_storage(Tensor::from_vec(base.clone(), &[3, 8], dev()).storage());
        let tol = 3e-2 * 1f32.max(num.abs());
        assert!((analytic[j] - num).abs() <= tol, "weight {j}: analytic {} vs numeric {num}", analytic[j]);
    }
}
