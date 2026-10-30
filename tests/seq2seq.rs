//! Encoder-decoder Transformer: cached decoding, padding masks, training.
//! Set `RUSTNN_DEVICE=mps` to run on the GPU.

use rustnn::functional as F;
use rustnn::nn::seq2seq::{Seq2Seq, Seq2SeqConfig};
use rustnn::nn::{KvCache, Module};
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

const PAD: usize = 0;
const BOS: usize = 1;
const EOS: usize = 2;

fn cfg() -> Seq2SeqConfig {
    Seq2SeqConfig {
        src_vocab: 10,
        tgt_vocab: 10,
        dim: 32,
        heads: 4,
        encoder_layers: 2,
        decoder_layers: 2,
        ff_hidden: 64,
        max_len: 12,
        dropout: 0.0,
    }
}

fn ids(v: &[usize], b: usize) -> Tensor {
    Tensor::from_indices(v, &[b, v.len() / b], dev())
}

#[test]
fn cached_decoding_matches_teacher_forcing() {
    let _l = lock();
    rand::manual_seed(0);
    let m = Seq2Seq::new(cfg(), dev());
    m.eval();
    let src = ids(&[3, 4, 5, 6, 7], 1);
    let tgt: Vec<usize> = vec![BOS, 7, 6, 5, 4, 3];
    no_grad(|| {
        let memory = m.encode(&src, None);
        let full = m.decode(&ids(&tgt, 1), &memory, None);
        let v = full.dim(-1);
        let mut cache = KvCache::default();
        let mut pos = 0;
        for c in [2, 1, 3] {
            let out = m.decode_cached(&ids(&tgt[pos..pos + c], 1), &memory, None, &mut cache);
            let want = full.narrow(1, pos, c).reshape(&[c * v]).to_vec();
            assert!(max_abs_diff(&out.reshape(&[c * v]).to_vec(), &want) < 1e-4);
            pos += c;
        }
        assert_eq!(cache.len(), 6);
    });
}

#[test]
fn source_padding_is_invisible_with_a_mask() {
    let _l = lock();
    rand::manual_seed(1);
    let m = Seq2Seq::new(cfg(), dev());
    m.eval();
    let tgt = ids(&[BOS, 5, 4], 1);
    no_grad(|| {
        let plain = m.forward_pair(&ids(&[3, 4, 5], 1), &tgt, None).to_vec();
        let padded_src = ids(&[3, 4, 5, PAD, PAD], 1);
        let mask = F::padding_mask(&padded_src, PAD);
        assert_eq!(mask.shape(), &[1, 1, 1, 5]);
        let masked = m.forward_pair(&padded_src, &tgt, Some(&mask)).to_vec();
        assert!(max_abs_diff(&plain, &masked) < 1e-4, "padding changed the output");
        let unmasked = m.forward_pair(&padded_src, &tgt, None).to_vec();
        assert!(max_abs_diff(&plain, &unmasked) > 1e-3, "without the mask the padding should matter");
    });
}

#[test]
fn cross_entropy_ignore_skips_rows() {
    let logits = Tensor::from_vec(vec![1., 2., 0.5, 0.1, -1., 3., 2., 2., 2.], &[3, 3], dev());
    let a = F::cross_entropy_ignore(&logits, &[2usize, 0, 1], 0).item();
    let b = F::cross_entropy(&logits.narrow(0, 0, 1), &[2usize], 0.0).item();
    let c = F::cross_entropy(&logits.narrow(0, 2, 1), &[1usize], 0.0).item();
    assert!((a - (b + c) / 2.0).abs() < 1e-5);
}

/// Reverse a random sequence of 3-5 tokens from 3..10 (batches are right-padded).
fn reverse_batch(n: usize) -> (Tensor, Tensor, Vec<usize>) {
    let (ls, lt) = (5, 7);
    let (mut src, mut tin, mut tout) = (vec![PAD; n * ls], vec![PAD; n * lt], vec![PAD; n * (lt)]);
    for i in 0..n {
        let len = 3 + rand::randint(3);
        let s: Vec<usize> = (0..len).map(|_| 3 + rand::randint(7)).collect();
        src[i * ls..i * ls + len].copy_from_slice(&s);
        let mut r: Vec<usize> = s.iter().rev().copied().collect();
        r.push(EOS);
        tin[i * lt] = BOS;
        tin[i * lt + 1..i * lt + 1 + len].copy_from_slice(&r[..len]);
        tout[i * lt..i * lt + len + 1].copy_from_slice(&r);
    }
    (ids(&src, n), ids(&tin, n), tout)
}

#[test]
fn learns_to_reverse() {
    let _l = lock();
    rand::manual_seed(2);
    let m = Seq2Seq::new(cfg(), dev());
    let mut opt = Adam::new(m.parameters(), 3e-3);
    for _ in 0..300 {
        let (src, tin, tout) = reverse_batch(32);
        let mask = F::padding_mask(&src, PAD);
        opt.zero_grad();
        let logits = m.forward_pair(&src, &tin, Some(&mask));
        let (b, t, v) = (logits.dim(0), logits.dim(1), logits.dim(2));
        F::cross_entropy_ignore(&logits.reshape(&[b * t, v]), &tout, PAD).backward();
        opt.step();
    }
    m.eval();
    let mut correct = 0;
    for _ in 0..20 {
        let len = 3 + rand::randint(3);
        let s: Vec<usize> = (0..len).map(|_| 3 + rand::randint(7)).collect();
        let mut want: Vec<usize> = s.iter().rev().copied().collect();
        want.push(EOS);
        correct += (m.greedy(&s, BOS, Some(EOS), 10) == want) as usize;
    }
    assert!(correct >= 18, "{correct}/20 sequences reversed");
}
