//! Preallocated KV cache: in-place appends into doubling buffers, views for
//! attention, copy-on-write clones, gradient fallback.
//! Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::nn::{KvCache, LayerKv};
use ruai::{no_grad, rand, Device, Tensor};

fn dev() -> Device {
    Device::from_env()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn kv(t: usize, seed: f32) -> Tensor {
    Tensor::from_vec((0..2 * 3 * t * 4).map(|i| i as f32 + seed).collect(), &[2, 3, t, 4], dev())
}

#[test]
fn appends_grow_by_doubling_and_return_views() {
    let mut c = LayerKv::default();
    let mut all_k: Vec<Tensor> = Vec::new();
    let mut caps = Vec::new();
    for step in 0..40 {
        let t = if step == 0 { 5 } else { 1 };
        let (k, v) = (kv(t, step as f32 * 1000.0), kv(t, -(step as f32) * 1000.0));
        all_k.push(k.clone());
        let (ck, cv) = c.append(k, v);
        assert_eq!(ck.shape(), &[2, 3, c.len(), 4]);
        assert_eq!(cv.shape()[2], c.len());
        if c.capacity() > c.len() {
            assert!(!ck.is_contiguous(), "a view of the larger buffer, not a copy");
        }
        caps.push(c.capacity());
    }
    assert_eq!(c.len(), 44);
    caps.dedup();
    assert_eq!(caps, vec![16, 32, 64], "capacity doubles");
    let want = Tensor::cat(&all_k.iter().collect::<Vec<_>>(), 2).to_vec();
    assert_eq!(c.keys().unwrap().to_vec(), want);
}

#[test]
fn earlier_views_and_clones_are_unaffected() {
    let mut c = LayerKv::default();
    let (k1, _) = c.append(kv(2, 0.0), kv(2, 0.0));
    let snapshot = k1.to_vec();
    let mut fork = c.clone();
    c.append(kv(1, 100.0), kv(1, 100.0));
    fork.append(kv(1, 200.0), kv(1, 200.0));
    assert_eq!(k1.to_vec(), snapshot, "a view returned earlier keeps its contents");
    let (a, b) = (c.keys().unwrap().to_vec(), fork.keys().unwrap().to_vec());
    assert_ne!(a, b, "a cloned cache is independent (copy on write)");
    assert_eq!(&a[..8], &snapshot[..8]);
    assert_eq!(&b[..8], &snapshot[..8]);
    // clear keeps the buffers
    let cap = c.capacity();
    c.clear();
    assert!(c.is_empty() && c.keys().is_none());
    c.append(kv(3, 7.0), kv(3, 7.0));
    assert_eq!((c.len(), c.capacity()), (3, cap));
    assert_eq!(c.keys().unwrap().to_vec(), kv(3, 7.0).to_vec());
}

#[test]
fn gradients_flow_through_the_cache() {
    let mut c = LayerKv::default();
    let a = kv(2, 0.0).requires_grad_();
    let b = kv(1, 5.0).requires_grad_();
    c.append(a.clone(), a.clone());
    let (k, _) = c.append(b.clone(), b.clone());
    (&k * 2.0).sum_all().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![2.0; a.numel()]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![2.0; b.numel()]);
    // later appends without gradients move into the buffers
    let (k, _) = no_grad(|| c.append(kv(1, 9.0), kv(1, 9.0)));
    assert_eq!(k.shape()[2], 4);
    assert!(!k.requires_grad());
}

#[test]
fn long_generation_matches_full_forward() {
    rand::manual_seed(5);
    let m = Llama::new(LlamaConfig::new(31, 32, 2, 4, 2, 128), dev());
    let ids: Vec<usize> = (0..100).map(|i| (i * 13 + 7) % 31).collect();
    no_grad(|| {
        let full = m.forward_tokens(&Tensor::from_indices(&ids, &[1, 100], dev()));
        let v = full.dim(-1);
        let mut cache = KvCache::default();
        m.forward_cached(&Tensor::from_indices(&ids[..10], &[1, 10], dev()), &mut cache);
        for (pos, &id) in ids.iter().enumerate().skip(10) {
            let out = m.forward_cached(&Tensor::from_indices(&[id], &[1, 1], dev()), &mut cache);
            let want = full.narrow(1, pos, 1).reshape(&[v]).to_vec();
            assert!(max_abs_diff(&out.reshape(&[v]).to_vec(), &want) < 1e-4, "position {pos}");
        }
        assert_eq!(cache.len(), 100);
        assert_eq!(cache.layers[0].capacity(), 128);
    });
}
