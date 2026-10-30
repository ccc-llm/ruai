//! DataLoader, datasets, LR schedules and metrics.

use ruai::data::{random_split, ClassificationData, DataLoader, Dataset, VecDataset};
use ruai::metrics::{Accuracy, Ema, RunningMean};
use ruai::optim::lr::{ConstantLr, CosineLr, ExponentialLr, LinearLr, LrScheduler, StepLr};
use ruai::{Device, Tensor};
use std::collections::HashSet;

fn ids(n: usize) -> VecDataset<usize> {
    VecDataset((0..n).collect())
}

fn epoch_batches(loader: &DataLoader<VecDataset<usize>, Vec<usize>>, e: u64) -> Vec<Vec<usize>> {
    loader.epoch(e).collect()
}

#[test]
fn sequential_batches_and_drop_last() {
    let l = DataLoader::new(ids(10), 4, |v: Vec<usize>| v);
    assert_eq!(l.len(), 3);
    assert_eq!(epoch_batches(&l, 0), vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]);
    let l = DataLoader::new(ids(10), 4, |v: Vec<usize>| v).drop_last(true);
    assert_eq!(l.len(), 2);
    assert_eq!(epoch_batches(&l, 0).len(), 2);
    assert_eq!(l.iter().len(), 2); // ExactSizeIterator
}

#[test]
fn shuffle_is_a_permutation_deterministic_and_per_epoch() {
    let l = DataLoader::new(ids(103), 8, |v: Vec<usize>| v).shuffle(true).seed(42);
    let e0: Vec<usize> = epoch_batches(&l, 0).concat();
    let mut sorted = e0.clone();
    sorted.sort();
    assert_eq!(sorted, (0..103).collect::<Vec<_>>());
    assert_eq!(e0, epoch_batches(&l, 0).concat(), "same epoch, same order");
    assert_ne!(e0, epoch_batches(&l, 1).concat(), "different epochs reshuffle");
    let other_seed = DataLoader::new(ids(103), 8, |v: Vec<usize>| v).shuffle(true).seed(7);
    assert_ne!(e0, epoch_batches(&other_seed, 0).concat());
}

#[test]
fn workers_produce_the_same_batches_in_order() {
    let base = epoch_batches(&DataLoader::new(ids(57), 5, |v: Vec<usize>| v).shuffle(true).seed(3), 2);
    for workers in [1, 2, 3, 8, 32] {
        let l = DataLoader::new(ids(57), 5, |v: Vec<usize>| v).shuffle(true).seed(3).num_workers(workers).prefetch(1);
        assert_eq!(epoch_batches(&l, 2), base, "num_workers = {workers}");
    }
}

#[test]
fn workers_actually_run_in_parallel() {
    // each batch sleeps 30 ms; 8 batches on 4 workers should take ~2 rounds, not 8
    let slow = |v: Vec<usize>| {
        std::thread::sleep(std::time::Duration::from_millis(30));
        v
    };
    let l = DataLoader::new(ids(8), 1, slow).num_workers(4);
    let t = std::time::Instant::now();
    assert_eq!(l.iter().count(), 8);
    assert!(t.elapsed().as_millis() < 8 * 30, "took {:?}", t.elapsed());
}

#[test]
fn dropping_an_epoch_early_does_not_hang() {
    let l = DataLoader::new(ids(1000), 1, |v: Vec<usize>| v).num_workers(4).prefetch(1);
    for (i, _) in l.iter().enumerate() {
        if i == 3 {
            break;
        }
    }
    // reaching here means the workers were unblocked and joined
}

#[test]
#[should_panic(expected = "bad sample")]
fn worker_panics_propagate() {
    let l = DataLoader::new(ids(10), 2, |v: Vec<usize>| {
        if v.contains(&5) {
            panic!("bad sample");
        }
        v
    })
    .num_workers(2);
    for _ in l.iter() {}
}

#[test]
fn classification_data_collates_to_tensors() {
    let feats: Vec<f32> = (0..6 * 2 * 3).map(|i| i as f32).collect();
    let data = ClassificationData::new(feats, vec![0, 1, 2, 0, 1, 2], &[2, 3]);
    let l = DataLoader::new(data, 4, ClassificationData::collate(Device::Cpu)).num_workers(2);
    let batches: Vec<_> = l.iter().collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].inputs.shape(), &[4, 6]);
    assert_eq!(batches[0].targets.to_vec_i32(), vec![0, 1, 2, 0]);
    assert_eq!(batches[1].inputs.to_vec()[..6], [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]);
    // a batch reshapes into the sample shape
    assert_eq!(batches[0].inputs.reshape(&[4, 2, 3]).shape(), &[4, 2, 3]);
}

#[test]
fn random_split_partitions() {
    let parts = random_split(ids(50), &[35, 10, 5], 1);
    let all: Vec<usize> = parts.iter().flat_map(|p| (0..p.len()).map(|i| p.get(i))).collect();
    assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), vec![35, 10, 5]);
    assert_eq!(all.iter().collect::<HashSet<_>>().len(), 50);
}

#[test]
fn lr_schedules() {
    let close = |a: f32, b: f32| assert!((a - b).abs() < 1e-7, "{a} vs {b}");
    close(ConstantLr(0.1).lr_at(1000), 0.1);
    let s = StepLr { base_lr: 1.0, step_size: 10, gamma: 0.5 };
    close(s.lr_at(9), 1.0);
    close(s.lr_at(10), 0.5);
    close(s.lr_at(25), 0.25);
    close(ExponentialLr { base_lr: 2.0, gamma: 0.5 }.lr_at(3), 0.25);
    let c = CosineLr { base_lr: 1.0, min_lr: 0.0, warmup: 4, total: 104 };
    close(c.lr_at(0), 0.25);
    close(c.lr_at(3), 1.0);
    close(c.lr_at(54), 0.5);
    close(c.lr_at(104), 0.0);
    close(c.lr_at(500), 0.0);
    let l = LinearLr { base_lr: 1.0, min_lr: 0.1, warmup: 0, total: 10 };
    close(l.lr_at(0), 1.0);
    close(l.lr_at(5), 0.55);
    close(l.lr_at(10), 0.1);
    // apply() drives an optimizer
    let p = Tensor::zeros(&[1], Device::Cpu).requires_grad_();
    let mut opt = ruai::optim::SGD::new(vec![p], 0.0);
    assert_eq!(s.apply(&mut opt, 20), 0.25);
    use ruai::optim::Optimizer;
    close(opt.lr(), 0.25);
}

#[test]
fn metrics() {
    let mut m = RunningMean::new();
    m.add_weighted(1.0, 3.0);
    m.add(5.0);
    assert_eq!(m.mean(), 2.0);
    let mut e = Ema::new(0.5);
    e.update(4.0);
    assert_eq!(e.update(0.0), 2.0);
    let mut acc = Accuracy::new();
    let logits = Tensor::from_slice(&[0.9, 0.1, 0.2, 0.8, 0.6, 0.4], &[3, 2]);
    acc.update(&logits, &[0usize, 1, 1]);
    acc.update(&logits.narrow(0, 0, 1), &[1usize]);
    assert_eq!(acc.value(), 0.5);
}
