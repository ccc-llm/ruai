//! Cross-device consistency: every backend primitive on each GPU backend must
//! match the CPU backend. Runs for each device compiled in and available.
//!
//!     cargo test --features mps --test devices

use ruai::backend::{backend_for, AdamParams, AttentionSpec, Backend, BinaryOp, Conv2dGeom, MatmulSpec, ReduceOp, Storage, UnaryOp};
use ruai::{rand, DType, Device};

fn gpu_devices() -> Vec<Device> {
    [Device::Mps, Device::Cuda(0)].into_iter().filter(|d| d.is_available()).collect()
}

struct Pair {
    cpu: &'static dyn Backend,
    gpu: &'static dyn Backend,
}

impl Pair {
    /// Same random f32 data on both devices.
    fn randn(&self, n: usize) -> (Storage, Storage) {
        let v = rand::normal_vec(n, 0.0, 1.0);
        (self.cpu.from_f32(v.clone()), self.gpu.from_f32(v))
    }

    fn ints(&self, v: Vec<i32>) -> (Storage, Storage) {
        (self.cpu.from_i32(v.clone()), self.gpu.from_i32(v))
    }
}

#[track_caller]
fn same(what: &str, cpu: &Storage, gpu: &Storage, tol: f32) {
    assert_eq!(cpu.dtype(), gpu.dtype(), "{what}: dtype");
    let (a, b) = (cpu.to_vec(), gpu.to_vec());
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let ok = (x - y).abs() <= tol * (1.0 + x.abs()) || (x.is_nan() && y.is_nan()) || x == y;
        assert!(ok, "{what}: elem {i}: cpu {x} vs gpu {y}");
    }
}

fn for_each_gpu(f: impl Fn(&Pair, Device)) {
    let devs = gpu_devices();
    if devs.is_empty() {
        eprintln!("no GPU backend available; skipping");
    }
    for d in devs {
        f(&Pair { cpu: backend_for(Device::Cpu), gpu: backend_for(d) }, d);
    }
}

#[test]
fn memory_and_casts() {
    for_each_gpu(|p, d| {
        let (c, g) = p.randn(1000);
        assert_eq!(g.device(), d);
        same("roundtrip", &c, &g, 0.0);
        same("full f32", &p.cpu.full(77, 2.5, DType::F32), &p.gpu.full(77, 2.5, DType::F32), 0.0);
        same("full i32", &p.cpu.full(5, 3.0, DType::I32), &p.gpu.full(5, 3.0, DType::I32), 0.0);
        let s = p.cpu.from_f32(vec![-2.7, 0.5, 3.9]);
        let sg = p.gpu.from_f32(vec![-2.7, 0.5, 3.9]);
        same("cast f32->i32", &p.cpu.cast(&s, DType::I32), &p.gpu.cast(&sg, DType::I32), 0.0);
        let (ci, gi) = p.ints(vec![-3, 0, 7]);
        same("cast i32->f32", &p.cpu.cast(&ci, DType::F32), &p.gpu.cast(&gi, DType::F32), 0.0);
        assert_eq!(p.gpu.to_i32(&gi), vec![-3, 0, 7]);
        // empty tensors
        same("empty", &p.cpu.from_f32(vec![]), &p.gpu.from_f32(vec![]), 0.0);
    });
}

#[test]
fn elementwise() {
    rand::manual_seed(1);
    for_each_gpu(|p, _| {
        let (c, g) = p.randn(3001);
        let ops = [
            UnaryOp::Neg,
            UnaryOp::Exp,
            UnaryOp::Abs,
            UnaryOp::Tanh,
            UnaryOp::Sigmoid,
            UnaryOp::Relu,
            UnaryOp::Gelu,
            UnaryOp::GeluGrad,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Recip,
        ];
        for op in ops {
            same(&format!("{op:?}"), &p.cpu.unary(op, &c), &p.gpu.unary(op, &g), 1e-5);
        }
        let ca = p.cpu.unary(UnaryOp::Abs, &c);
        let ga = p.gpu.unary(UnaryOp::Abs, &g);
        for op in [UnaryOp::Log, UnaryOp::Sqrt] {
            same(&format!("{op:?}"), &p.cpu.unary(op, &ca), &p.gpu.unary(op, &ga), 1e-5);
        }

        // broadcasting binary ops over several shape pairs
        let cases: [(&[usize], &[usize]); 5] =
            [(&[7, 9], &[7, 9]), (&[7, 9], &[9]), (&[2, 1, 5], &[3, 1]), (&[4, 6], &[]), (&[2, 3, 4, 5], &[1, 3, 1, 5])];
        for (sa, sb) in cases {
            let out = ruai::shape::broadcast_shape(sa, sb);
            let (ac, ag) = p.randn(sa.iter().product());
            let (bc, bg) = p.randn(sb.iter().product());
            let ops = [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Maximum,
                BinaryOp::Minimum,
                BinaryOp::Gt,
                BinaryOp::Lt,
                BinaryOp::Eq,
            ];
            for op in ops {
                same(
                    &format!("{op:?} {sa:?} {sb:?}"),
                    &p.cpu.binary(op, &ac, sa, &bc, sb, &out),
                    &p.gpu.binary(op, &ag, sa, &bg, sb, &out),
                    1e-5,
                );
            }
            same(
                &format!("expand {sb:?}->{out:?}"),
                &p.cpu.expand(&bc, sb, &out),
                &p.gpu.expand(&bg, sb, &out),
                0.0,
            );
        }
        // comparisons on integer inputs
        let (ic, ig) = p.ints(vec![1, 2, 3, 4]);
        let (jc, jg) = p.ints(vec![1, 0, 3, 9]);
        same("eq i32", &p.cpu.binary(BinaryOp::Eq, &ic, &[4], &jc, &[4], &[4]), &p.gpu.binary(BinaryOp::Eq, &ig, &[4], &jg, &[4], &[4]), 0.0);
        // where
        let (cc, cg) = (p.cpu.from_f32(vec![1., 0., 1., 0., 0.]), p.gpu.from_f32(vec![1., 0., 1., 0., 0.]));
        let (ac, ag) = p.randn(5);
        let (bc, bg) = p.randn(5);
        same("where", &p.cpu.where_cond(&cc, &ac, &bc), &p.gpu.where_cond(&cg, &ag, &bg), 0.0);
    });
}

#[test]
fn reductions() {
    rand::manual_seed(2);
    for_each_gpu(|p, _| {
        // includes rows longer than one threadgroup and odd sizes
        for shape in [vec![3, 1000], vec![5, 7, 3], vec![1, 100_000], vec![64, 33], vec![4, 1]] {
            let (c, g) = p.randn(shape.iter().product());
            for axis in 0..shape.len() {
                for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::ArgMax] {
                    // f32 summation error grows like sqrt(length) times the
                    // terms' size, independent of how small the (cancelled) sum is
                    let tol = if op == ReduceOp::Sum { 1e-4 * (shape[axis] as f32).sqrt() } else { 0.0 };
                    same(
                        &format!("{op:?} {shape:?} axis {axis}"),
                        &p.cpu.reduce(op, &c, &shape, axis),
                        &p.gpu.reduce(op, &g, &shape, axis),
                        tol,
                    );
                }
            }
        }
    });
}

#[test]
fn matmul() {
    rand::manual_seed(3);
    for_each_gpu(|p, _| {
        for &(batch, m, k, n) in &[(1, 1, 1, 1), (1, 17, 33, 9), (4, 64, 32, 48), (3, 5, 129, 7)] {
            for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                for (sa, sb) in [(m * k, k * n), (0, k * n), (m * k, 0)] {
                    let ablocks = if sa == 0 { 1 } else { batch };
                    let bblocks = if sb == 0 { 1 } else { batch };
                    let (ac, ag) = p.randn(ablocks * m * k);
                    let (bc, bg) = p.randn(bblocks * k * n);
                    let spec = MatmulSpec { batch, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: sa, b_batch_stride: sb };
                    same(
                        &format!("matmul {spec:?}"),
                        &p.cpu.matmul(&ac, &bc, spec),
                        &p.gpu.matmul(&ag, &bg, spec),
                        1e-4,
                    );
                }
            }
        }
    });
}

#[test]
fn layout_and_indexing() {
    rand::manual_seed(4);
    for_each_gpu(|p, _| {
        let shape = [2, 3, 4, 5];
        let (c, g) = p.randn(120);
        for perm in [[0, 1, 3, 2], [3, 2, 1, 0], [1, 0, 2, 3], [2, 3, 0, 1]] {
            same(&format!("permute {perm:?}"), &p.cpu.permute(&c, &shape, &perm), &p.gpu.permute(&g, &shape, &perm), 0.0);
        }
        for (axis, start, len) in [(0, 1, 1), (2, 1, 2), (3, 0, 5), (3, 4, 1)] {
            same(
                &format!("narrow {axis} {start} {len}"),
                &p.cpu.narrow(&c, &shape, axis, start, len),
                &p.gpu.narrow(&g, &shape, axis, start, len),
                0.0,
            );
        }
        let (c2, g2) = p.randn(2 * 3 * 2 * 5);
        let s2 = [2, 3, 2, 5];
        same(
            "cat axis 2",
            &p.cpu.cat(&[&c, &c2], &[&shape, &s2], 2),
            &p.gpu.cat(&[&g, &g2], &[&shape, &s2], 2),
            0.0,
        );
        let (ic, ig) = p.ints(vec![4, 0, 4, 2, 1, 4]);
        same("cat i32", &p.cpu.cat(&[&ic, &ic], &[&[6], &[6]], 0), &p.gpu.cat(&[&ig, &ig], &[&[6], &[6]], 0), 0.0);

        // embedding gather / scatter with repeated indices
        let (tc, tg) = p.randn(5 * 7);
        same("index_select", &p.cpu.index_select(&tc, 7, &ic), &p.gpu.index_select(&tg, 7, &ig), 0.0);
        let (sc, sg) = p.randn(6 * 7);
        same("index_add", &p.cpu.index_add(&sc, 5, 7, &ic), &p.gpu.index_add(&sg, 5, 7, &ig), 1e-5);

        // gather along the middle axis of [3, 4, 2] with idx [3, 5, 2]
        let xs = [3, 4, 2];
        let (xc, xg) = p.randn(24);
        let idx: Vec<i32> = (0..30).map(|i| i * 7 % 4).collect();
        let (jc, jg) = p.ints(idx);
        same("gather", &p.cpu.gather(&xc, &xs, 1, &jc, 5), &p.gpu.gather(&xg, &xs, 1, &jg, 5), 0.0);
        let (sc, sg) = p.randn(30);
        same("scatter_add", &p.cpu.scatter_add(&sc, &jc, 5, &xs, 1), &p.gpu.scatter_add(&sg, &jg, 5, &xs, 1), 1e-5);
    });
}

#[test]
fn convolution_lowering() {
    rand::manual_seed(5);
    for_each_gpu(|p, _| {
        for (kh, stride, padding) in [(3, 1, 1), (3, 2, 0), (2, 2, 0), (5, 1, 2), (1, 1, 0)] {
            let g = Conv2dGeom { n: 2, c: 3, h: 9, w: 7, kh, kw: kh, stride, padding };
            let (xc, xg) = p.randn(2 * 3 * 9 * 7);
            let cc = p.cpu.im2col(&xc, g);
            let cg = p.gpu.im2col(&xg, g);
            same(&format!("im2col {g:?}"), &cc, &cg, 0.0);
            let (rc, rg) = p.randn(cc.len());
            same(&format!("col2im {g:?}"), &p.cpu.col2im(&rc, g), &p.gpu.col2im(&rg, g), 1e-5);
        }
    });
}

#[test]
fn fused_ops() {
    rand::manual_seed(6);
    for_each_gpu(|p, _| {
        for shape in [vec![4, 10], vec![3, 1000], vec![2, 3, 257], vec![6, 1]] {
            let n = shape.iter().product();
            let (xc, xg) = p.randn(n);
            let (gc, gg) = p.randn(n);
            for axis in 0..shape.len() {
                let (yc, yg) = (p.cpu.softmax(&xc, &shape, axis), p.gpu.softmax(&xg, &shape, axis));
                same(&format!("softmax {shape:?}/{axis}"), &yc, &yg, 1e-5);
                same(
                    &format!("softmax_bwd {shape:?}/{axis}"),
                    &p.cpu.softmax_backward(&yc, &gc, &shape, axis),
                    &p.gpu.softmax_backward(&yg, &gg, &shape, axis),
                    1e-4,
                );
                let (yc, yg) = (p.cpu.log_softmax(&xc, &shape, axis), p.gpu.log_softmax(&xg, &shape, axis));
                same(&format!("log_softmax {shape:?}/{axis}"), &yc, &yg, 1e-5);
                same(
                    &format!("log_softmax_bwd {shape:?}/{axis}"),
                    &p.cpu.log_softmax_backward(&yc, &gc, &shape, axis),
                    &p.gpu.log_softmax_backward(&yg, &gg, &shape, axis),
                    1e-4,
                );
            }
            let dim = shape[shape.len() - 1];
            let rows = n / dim;
            let (xhc, rc) = p.cpu.layer_norm(&xc, rows, dim, 1e-5);
            let (xhg, rg) = p.gpu.layer_norm(&xg, rows, dim, 1e-5);
            same("layer_norm xhat", &xhc, &xhg, 1e-4);
            same("layer_norm rstd", &rc, &rg, 1e-4);
            same(
                "layer_norm_bwd",
                &p.cpu.layer_norm_backward(&gc, &xhc, &rc, rows, dim),
                &p.gpu.layer_norm_backward(&gg, &xhg, &rg, rows, dim),
                1e-4,
            );
        }
        let n = 5000;
        let (pc, pg) = p.randn(n);
        let (gc, gg) = p.randn(n);
        let (mc, mg) = p.randn(n);
        let v = rand::uniform_vec(n, 0.0, 1.0);
        let (vc, vg) = (p.cpu.from_f32(v.clone()), p.gpu.from_f32(v));
        let h = AdamParams {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decoupled: true,
            bias_correction1: 0.271,
            bias_correction2: 0.003,
        };
        let a = p.cpu.adam_step(&pc, &gc, &mc, &vc, h);
        let b = p.gpu.adam_step(&pg, &gg, &mg, &vg, h);
        same("adam p", &a.0, &b.0, 1e-5);
        same("adam m", &a.1, &b.1, 1e-5);
        same("adam v", &a.2, &b.2, 1e-5);
    });
}

#[test]
fn device_from_env_and_parse() {
    assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
    assert_eq!("mps".parse::<Device>().unwrap(), Device::Mps);
    assert_eq!("cuda:2".parse::<Device>().unwrap(), Device::Cuda(2));
    assert!("tpu".parse::<Device>().is_err());
    // tensors created via from_env really live there
    let d = Device::from_env();
    if d.is_available() {
        assert_eq!(ruai::Tensor::zeros(&[2], d).device(), d);
    }
}

/// Same random data as `dtype` on both devices.
fn randn_as(p: &Pair, n: usize, dtype: DType) -> (Storage, Storage) {
    let (c, g) = p.randn(n);
    (p.cpu.cast(&c, dtype), p.gpu.cast(&g, dtype))
}

/// Tolerance for comparing half-precision results: both sides compute in f32
/// and round once, so they may differ by about one unit in the last place.
fn half_tol(d: DType) -> f32 {
    if d == DType::BF16 {
        1e-2
    } else {
        2e-3
    }
}

#[test]
fn half_precision_primitives() {
    rand::manual_seed(7);
    for_each_gpu(|p, _| {
        for d in [DType::F16, DType::BF16] {
            let tol = half_tol(d);
            let (c, g) = randn_as(p, 1001, d);
            assert_eq!(g.dtype(), d);
            same(&format!("{d} cast"), &c, &g, 0.0);
            // every pairwise cast
            for to in [DType::F32, DType::F16, DType::BF16, DType::I32] {
                same(&format!("cast {d}->{to}"), &p.cpu.cast(&c, to), &p.gpu.cast(&g, to), 0.0);
            }
            same(&format!("{d} full"), &p.cpu.full(9, 1.5, d), &p.gpu.full(9, 1.5, d), 0.0);
            for op in [UnaryOp::Neg, UnaryOp::Exp, UnaryOp::Tanh, UnaryOp::Gelu, UnaryOp::Relu, UnaryOp::Sigmoid] {
                same(&format!("{d} {op:?}"), &p.cpu.unary(op, &c), &p.gpu.unary(op, &g), tol);
            }
            let (sa, sb, so): (&[usize], &[usize], &[usize]) = (&[7, 11], &[11], &[7, 11]);
            let (ac, ag) = randn_as(p, 77, d);
            let (bc, bg) = randn_as(p, 11, d);
            for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Maximum, BinaryOp::Gt] {
                let (x, y) = (p.cpu.binary(op, &ac, sa, &bc, sb, so), p.gpu.binary(op, &ag, sa, &bg, sb, so));
                assert_eq!(y.dtype(), d, "{op:?} keeps the dtype");
                same(&format!("{d} {op:?}"), &x, &y, tol);
            }
            let shape = [3, 5, 7];
            let (xc, xg) = randn_as(p, 105, d);
            for axis in 0..3 {
                for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::ArgMax] {
                    same(
                        &format!("{d} {op:?} axis {axis}"),
                        &p.cpu.reduce(op, &xc, &shape, axis),
                        &p.gpu.reduce(op, &xg, &shape, axis),
                        tol,
                    );
                }
            }
            same(&format!("{d} permute"), &p.cpu.permute(&xc, &shape, &[2, 0, 1]), &p.gpu.permute(&xg, &shape, &[2, 0, 1]), 0.0);
            same(&format!("{d} narrow"), &p.cpu.narrow(&xc, &shape, 1, 1, 3), &p.gpu.narrow(&xg, &shape, 1, 1, 3), 0.0);
            same(
                &format!("{d} cat"),
                &p.cpu.cat(&[&xc, &xc], &[&shape, &shape], 2),
                &p.gpu.cat(&[&xg, &xg], &[&shape, &shape], 2),
                0.0,
            );
            same(&format!("{d} expand"), &p.cpu.expand(&bc, sb, so), &p.gpu.expand(&bg, sb, so), 0.0);
            let (cc, cg) = (p.cpu.cast(&p.cpu.from_f32(vec![1., 0., 0., 1.]), d), p.gpu.cast(&p.gpu.from_f32(vec![1., 0., 0., 1.]), d));
            let (wc, wg) = randn_as(p, 4, d);
            let (vc, vg) = randn_as(p, 4, d);
            same(&format!("{d} where"), &p.cpu.where_cond(&cc, &wc, &vc), &p.gpu.where_cond(&cg, &wg, &vg), 0.0);
            let (ic, ig) = p.ints(vec![3, 0, 3, 1]);
            let (tc, tg) = randn_as(p, 4 * 6, d);
            same(&format!("{d} index_select"), &p.cpu.index_select(&tc, 6, &ic), &p.gpu.index_select(&tg, 6, &ig), 0.0);
            let geo = Conv2dGeom { n: 2, c: 3, h: 6, w: 5, kh: 3, kw: 3, stride: 1, padding: 1 };
            let (xc, xg) = randn_as(p, 2 * 3 * 6 * 5, d);
            let (colc, colg) = (p.cpu.im2col(&xc, geo), p.gpu.im2col(&xg, geo));
            same(&format!("{d} im2col"), &colc, &colg, 0.0);
            same(&format!("{d} col2im"), &p.cpu.col2im(&colc, geo), &p.gpu.col2im(&colg, geo), tol);
            // f32-only kernels run through the widening fallback
            let rs = [4, 33];
            let (sc, sg) = randn_as(p, 4 * 33, d);
            same(&format!("{d} softmax"), &p.cpu.softmax(&sc, &rs, 1), &p.gpu.softmax(&sg, &rs, 1), tol);
            let (lc, _) = p.cpu.layer_norm(&sc, 4, 33, 1e-5);
            let (lg, _) = p.gpu.layer_norm(&sg, 4, 33, 1e-5);
            same(&format!("{d} layer_norm"), &lc, &lg, tol * 2.0);
        }
    });
}

#[test]
fn half_precision_matmul() {
    rand::manual_seed(8);
    for_each_gpu(|p, _| {
        for d in [DType::F16, DType::BF16] {
            for &(batch, m, k, n) in &[(1, 17, 33, 9), (3, 16, 64, 24)] {
                for (ta, tb) in [(false, false), (true, false), (false, true)] {
                    for (sa, sb) in [(m * k, k * n), (0, k * n)] {
                        let ablocks = if sa == 0 { 1 } else { batch };
                        let (ac, ag) = randn_as(p, ablocks * m * k, d);
                        let (bc, bg) = randn_as(p, batch * k * n, d);
                        let spec = MatmulSpec { batch, m, k, n, trans_a: ta, trans_b: tb, a_batch_stride: sa, b_batch_stride: sb };
                        let (x, y) = (p.cpu.matmul(&ac, &bc, spec), p.gpu.matmul(&ag, &bg, spec));
                        assert_eq!(y.dtype(), d);
                        // f32 accumulation on both sides; GPU summation order differs
                        same(&format!("{d} matmul {spec:?}"), &x, &y, half_tol(d) * 3.0);
                    }
                }
            }
        }
    });
}

#[test]
fn int8_matmul() {
    rand::manual_seed(9);
    for_each_gpu(|p, _| {
        for d in [DType::F32, DType::F16, DType::BF16] {
            // m <= 16 takes the fused kernel, larger m the dequantize + GEMM path
            for &(m, k, n) in &[(1, 64, 300), (5, 33, 70), (40, 32, 48)] {
                let (xc, xg) = randn_as(p, m * k, d);
                let q: Vec<i8> = (0..k * n).map(|i| ((i * 37 % 255) as i32 - 127) as i8).collect();
                let s: Vec<f32> = (0..n).map(|j| 0.001 + j as f32 * 1e-5).collect();
                let (qc, qg) = (p.cpu.from_i8(q.clone()), p.gpu.from_i8(q));
                let (sc, sg) = (p.cpu.from_f32(s.clone()), p.gpu.from_f32(s));
                assert_eq!(qg.dtype(), DType::I8);
                same("i8 roundtrip", &qc, &qg, 0.0);
                let tol = if d == DType::F32 { 1e-4 } else { half_tol(d) * 3.0 };
                same(
                    &format!("{d} matmul_int8 m={m}"),
                    &p.cpu.matmul_int8(&xc, &qc, &sc, m, k, n),
                    &p.gpu.matmul_int8(&xg, &qg, &sg, m, k, n),
                    tol,
                );
            }
        }
        // i8 casts and copies
        let (ic, ig) = (p.cpu.from_i8(vec![-128, -1, 0, 7, 127]), p.gpu.from_i8(vec![-128, -1, 0, 7, 127]));
        for to in [DType::F32, DType::F16, DType::BF16, DType::I32] {
            same(&format!("cast i8->{to}"), &p.cpu.cast(&ic, to), &p.gpu.cast(&ig, to), 0.0);
        }
        let (fc, fg) = (p.cpu.from_f32(vec![-300.0, -2.7, 0.0, 3.9, 500.0]), p.gpu.from_f32(vec![-300.0, -2.7, 0.0, 3.9, 500.0]));
        same("cast f32->i8 (saturating)", &p.cpu.cast(&fc, DType::I8), &p.gpu.cast(&fg, DType::I8), 0.0);
        same("i8 narrow", &p.cpu.narrow(&ic, &[5], 0, 1, 3), &p.gpu.narrow(&ig, &[5], 0, 1, 3), 0.0);
        same("i8 full", &p.cpu.full(4, -3.0, DType::I8), &p.gpu.full(4, -3.0, DType::I8), 0.0);
    });
}

#[test]
fn int4_matmul() {
    rand::manual_seed(10);
    for_each_gpu(|p, _| {
        for d in [DType::F32, DType::F16, DType::BF16] {
            // m <= 16: fused kernel; larger m: dequantize + GEMM
            for &(m, k, n, group) in &[(1, 64, 300, 32), (5, 32, 70, 8), (40, 64, 48, 16)] {
                let (xc, xg) = randn_as(p, m * k, d);
                let q: Vec<i8> = (0..k / 2 * n).map(|i| (i * 73 % 256) as u8 as i8).collect();
                let s: Vec<f32> = (0..k / group * n).map(|j| 0.01 + (j % 13) as f32 * 1e-3).collect();
                let (qc, qg) = (p.cpu.from_i8(q.clone()), p.gpu.from_i8(q));
                let (sc, sg) = (p.cpu.from_f32(s.clone()), p.gpu.from_f32(s));
                same("dequant_int4", &p.cpu.dequant_int4(&qc, &sc, k, n, group), &p.gpu.dequant_int4(&qg, &sg, k, n, group), 0.0);
                let tol = if d == DType::F32 { 1e-4 } else { half_tol(d) * 3.0 };
                same(
                    &format!("{d} matmul_int4 m={m}"),
                    &p.cpu.matmul_int4(&xc, &qc, &sc, m, k, n, group),
                    &p.gpu.matmul_int4(&xg, &qg, &sg, m, k, n, group),
                    tol,
                );
            }
        }
    });
}

#[test]
fn fused_attention() {
    rand::manual_seed(11);
    for_each_gpu(|p, _| {
        for d in [DType::F32, DType::F16, DType::BF16] {
            for &(b, h, tq, tk, dk, dv, mask) in &[
                (2, 3, 5, 5, 16, 16, Some([1, 1, 5, 5])),
                (1, 2, 1, 9, 64, 32, None),
                (2, 2, 4, 6, 40, 40, Some([2, 1, 1, 6])),
                (1, 1, 3, 70, 256, 256, Some([1, 1, 3, 70])),
            ] {
                let (qc, qg) = randn_as(p, b * h * tq * dk, d);
                let (kc, kg) = randn_as(p, b * h * tk * dk, d);
                let (vc, vg) = randn_as(p, b * h * tk * dv, d);
                let spec = AttentionSpec { b, h, tq, tk, d: dk, dv, scale: 1.0 / (dk as f32).sqrt(), mask_shape: mask };
                let m = mask.map(|ms: [usize; 4]| p.randn(ms.iter().product()));
                let tol = if d == DType::F32 { 1e-4 } else { half_tol(d) * 3.0 };
                same(
                    &format!("{d} attention b={b} h={h} tq={tq} tk={tk} d={dk} dv={dv}"),
                    &p.cpu.attention(&qc, &kc, &vc, m.as_ref().map(|m| &m.0), spec),
                    &p.gpu.attention(&qg, &kg, &vg, m.as_ref().map(|m| &m.1), spec),
                    tol,
                );
            }
        }
    });
}

#[test]
fn strided_copy_all_dtypes() {
    rand::manual_seed(12);
    for_each_gpu(|p, _| {
        let layouts: [(&[usize], &[usize], usize); 4] = [
            (&[4, 2, 3], &[1, 12, 4], 0),
            (&[2, 3, 2], &[12, 4, 1], 1),
            (&[2, 3, 4], &[0, 1, 0], 2),
            (&[3, 5], &[1, 3], 4),
        ];
        let ints: Vec<i32> = (0..40).map(|i| i * 3 - 50).collect();
        for d in [DType::F32, DType::F16, DType::BF16, DType::I32, DType::I8] {
            let (c, g) = if d == DType::I32 || d == DType::I8 {
                let (c, g) = p.ints(ints.clone());
                (p.cpu.cast(&c, d), p.gpu.cast(&g, d))
            } else {
                let (c, g) = p.randn(40);
                (p.cpu.cast(&c, d), p.gpu.cast(&g, d))
            };
            for (shape, strides, off) in layouts {
                same(&format!("{d} strided_copy {shape:?} {strides:?}+{off}"), &p.cpu.strided_copy(&c, shape, strides, off), &p.gpu.strided_copy(&g, shape, strides, off), 0.0);
            }
        }
    });
}
