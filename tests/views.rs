//! Strided views: permute / transpose / narrow / select / expand /
//! unsqueeze / squeeze / reshape share the buffer; contiguous copies happen
//! only when a kernel needs one. Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::{rand, DType, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Host reference for x[i0, i1, i2] of a row-major [2, 3, 4] tensor of 0..24.
fn at(i: usize, j: usize, k: usize) -> f32 {
    (i * 12 + j * 4 + k) as f32
}

#[test]
fn view_ops_share_the_buffer() {
    let x = Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[2, 3, 4], dev());
    let p = x.permute(&[2, 0, 1]); // [4, 2, 3]
    assert!(!p.is_contiguous());
    let (buf, strides, off) = p.strided_parts();
    assert_eq!((buf.len(), strides, off), (24, vec![1, 12, 4], 0), "no copy, permuted strides");
    let want: Vec<f32> = (0..4).flat_map(|k| (0..2).flat_map(move |i| (0..3).map(move |j| at(i, j, k)))).collect();
    assert_eq!(p.to_vec(), want);

    let n = x.narrow(2, 1, 2); // [2, 3, 2]
    assert_eq!(n.strided_parts().2, 1, "narrow only moves the offset");
    let want: Vec<f32> = (0..2).flat_map(|i| (0..3).flat_map(move |j| (1..3).map(move |k| at(i, j, k)))).collect();
    assert_eq!(n.to_vec(), want);

    let s = x.select(1, 2); // [2, 4]
    assert_eq!(s.to_vec(), (0..2).flat_map(|i| (0..4).map(move |k| at(i, 2, k))).collect::<Vec<_>>());

    let e = Tensor::from_vec(vec![1., 2., 3.], &[3, 1], dev()).expand(&[2, 3, 4]);
    assert_eq!(e.strided_parts().1, vec![0, 1, 0], "broadcast dims have stride 0");
    assert_eq!(e.to_vec(), (0..24).map(|i| ((i / 4) % 3 + 1) as f32).collect::<Vec<_>>());

    let u = x.unsqueeze(1).squeeze(1);
    assert!(u.is_contiguous());
    assert_eq!(u.to_vec(), x.to_vec());

    // reshape of a row-major view is a view; of a permuted view a copy
    assert_eq!(x.narrow(0, 1, 1).reshape(&[12]).strided_parts().2, 12);
    assert_eq!(p.reshape(&[24]).to_vec(), p.to_vec());
    assert!(p.contiguous().is_contiguous());
}

#[test]
fn chains_of_views_match_the_host() {
    rand::manual_seed(0);
    let v = rand::normal_vec(2 * 3 * 4 * 5, 0.0, 1.0);
    let x = Tensor::from_vec(v.clone(), &[2, 3, 4, 5], dev());
    // y[a, b, c] = x[1, c, a + 1, b] for a < 3, b < 5, c < 3
    let y = x.select(0, 1).permute(&[1, 2, 0]).narrow(0, 1, 3);
    assert_eq!(y.shape(), &[3, 5, 3]);
    let mut want = Vec::new();
    for a in 0..3 {
        for b in 0..5 {
            for c in 0..3 {
                want.push(v[60 + c * 20 + (a + 1) * 5 + b]);
            }
        }
    }
    assert_eq!(y.to_vec(), want);
    // arithmetic on views
    let z = &y * 2.0 + &y.narrow(2, 0, 1);
    let want2: Vec<f32> = want.chunks(3).flat_map(|r| r.iter().map(move |&w| 2.0 * w + r[0]).collect::<Vec<_>>()).collect();
    assert!(max_abs_diff(&z.to_vec(), &want2) < 1e-6);
}

#[test]
fn views_are_snapshots() {
    // replacing a tensor's data (as optimizers do) never changes existing views
    let w = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2], dev());
    let wt = w.t();
    w.set_storage(Tensor::from_vec(vec![9., 9., 9., 9.], &[2, 2], dev()).storage());
    assert_eq!(wt.to_vec(), vec![1., 3., 2., 4.]);
    assert_eq!(w.t().to_vec(), vec![9., 9., 9., 9.]);
}

#[test]
fn matmul_reads_views_in_place() {
    rand::manual_seed(1);
    let a = Tensor::randn(&[2, 3, 4, 5], dev());
    let b = Tensor::randn(&[2, 3, 6, 5], dev());
    let want = a.matmul(&b.t().contiguous()).to_vec();
    assert!(max_abs_diff(&a.matmul(&b.t()).to_vec(), &want) < 1e-4, "transposed view");
    // batch-strided: rows 0..4 of a [2, 3, 7, 5] buffer, like a KV cache
    let big = Tensor::randn(&[2, 3, 7, 5], dev());
    let kv = big.narrow(2, 0, 4);
    assert!(!kv.is_contiguous());
    let q = Tensor::randn(&[2, 3, 2, 5], dev());
    let want = q.matmul(&kv.contiguous().t()).to_vec();
    assert!(max_abs_diff(&q.matmul(&kv.t()).to_vec(), &want) < 1e-4, "strided batch, transposed");
    let p = Tensor::randn(&[2, 3, 2, 4], dev());
    let want = p.matmul(&kv.contiguous()).to_vec();
    assert!(max_abs_diff(&p.matmul(&kv).to_vec(), &want) < 1e-4, "strided batch");
    // broadcast batch via expand (stride 0)
    let w = Tensor::randn(&[5, 3], dev());
    let we = w.unsqueeze(0).expand(&[4, 5, 3]);
    let x = Tensor::randn(&[4, 2, 5], dev());
    let want = x.matmul(&w).to_vec();
    assert!(max_abs_diff(&x.matmul(&we).to_vec(), &want) < 1e-4, "broadcast batch");
    // gradients through views
    let a = Tensor::randn(&[3, 4], dev()).requires_grad_();
    let b = Tensor::randn(&[5, 4], dev()).requires_grad_();
    a.matmul(&b.t()).sum_all().backward();
    let gb = b.grad().unwrap().to_vec();
    let col_sums: Vec<f32> = (0..4).map(|j| a.to_vec().chunks(4).map(|r| r[j]).sum()).collect();
    for r in gb.chunks(4) {
        assert!(max_abs_diff(r, &col_sums) < 1e-4);
    }
}

#[test]
fn views_of_every_dtype() {
    for d in [DType::F16, DType::BF16, DType::I32, DType::I8] {
        let x = Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[2, 3, 4], dev()).to_dtype(d);
        let p = x.permute(&[2, 0, 1]).narrow(1, 1, 1);
        assert_eq!(p.dtype(), d);
        let want: Vec<f32> = (0..4).flat_map(|k| (0..3).map(move |j| at(1, j, k))).collect();
        assert_eq!(p.to_vec(), want, "{d}");
    }
}
