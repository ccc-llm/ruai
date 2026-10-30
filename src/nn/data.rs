//! Datasets and a multi-threaded, prefetching `DataLoader` (like
//! `torch.utils.data`).
//!
//! * [`Dataset`]: random access to samples (`len` + `get`).
//! * [`DataLoader`]: groups shuffled indices into batches, turns each batch of
//!   samples into a value with a *collate* function (usually tensors), and
//!   builds batches ahead of time on `num_workers` background threads.
//!   Batch order is deterministic for a given `seed` and epoch, regardless
//!   of the number of workers.
//!
//! ```
//! use ruai::data::{ClassificationData, DataLoader};
//! use ruai::Device;
//! let data = ClassificationData::new(vec![0.0; 10 * 4], vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1], &[4]);
//! let loader = DataLoader::new(data, 3, ClassificationData::collate(Device::Cpu))
//!     .shuffle(true)
//!     .num_workers(2);
//! for batch in loader.epoch(0) {
//!     assert_eq!(batch.inputs.shape()[1], 4);
//! }
//! assert_eq!(loader.len(), 4); // 3 + 3 + 3 + 1
//! ```

use crate::nn::rand::Rng;
use crate::nn::tensor::Tensor;
use crate::nn::Device;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Random-access collection of samples.
pub trait Dataset: Send + Sync {
    /// Sample type returned by [`Dataset::get`].
    type Item: Send + 'static;

    /// Number of samples.
    fn len(&self) -> usize;

    /// Sample `index` (`0..len()`).
    fn get(&self, index: usize) -> Self::Item;

    /// `true` if there are no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A dataset backed by a `Vec` of samples.
///
/// (A wrapper rather than `impl Dataset for Vec<T>`, so that importing the
/// trait never shadows the slice method `Vec::get`.)
#[derive(Clone, Debug)]
pub struct VecDataset<T>(pub Vec<T>);

impl<T: Clone + Send + Sync + 'static> Dataset for VecDataset<T> {
    type Item = T;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> T {
        self.0[index].clone()
    }
}

impl<D: Dataset + ?Sized> Dataset for Arc<D> {
    type Item = D::Item;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn get(&self, index: usize) -> D::Item {
        (**self).get(index)
    }
}

/// A view of a dataset restricted to some indices.
pub struct Subset<D> {
    /// The underlying dataset.
    pub dataset: Arc<D>,
    /// Indices into `dataset`; sample `i` of the subset is `dataset.get(indices[i])`.
    pub indices: Vec<usize>,
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, index: usize) -> D::Item {
        self.dataset.get(self.indices[index])
    }
}

/// Randomly partition a dataset into subsets of the given sizes
/// (e.g. train / validation). Sizes must add up to `dataset.len()`.
pub fn random_split<D: Dataset>(dataset: D, lengths: &[usize], seed: u64) -> Vec<Subset<D>> {
    assert_eq!(lengths.iter().sum::<usize>(), dataset.len(), "random_split: lengths must sum to the dataset size");
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    Rng::new(seed).shuffle(&mut order);
    let ds = Arc::new(dataset);
    let mut start = 0;
    lengths
        .iter()
        .map(|&n| {
            let s = Subset { dataset: ds.clone(), indices: order[start..start + n].to_vec() };
            start += n;
            s
        })
        .collect()
}

/// In-memory classification data: `N` samples of `sample_shape` floats plus a label each.
#[derive(Clone)]
pub struct ClassificationData {
    /// Flattened features, `N * prod(sample_shape)` values.
    pub features: Vec<f32>,
    /// Class label per sample, length `N`.
    pub labels: Vec<usize>,
    /// Shape of one sample.
    pub sample_shape: Vec<usize>,
}

/// A batch of inputs and `I32` targets `[B]`. [`ClassificationData::collator`]
/// gives inputs `[B, ...sample_shape]`, [`ClassificationData::collate`] flattened `[B, prod(sample_shape)]`.
pub struct Batch {
    /// Input features.
    pub inputs: Tensor,
    /// `I32` class targets `[B]`.
    pub targets: Tensor,
}

impl ClassificationData {
    /// Wrap flattened `features` and `labels`; panics if `features.len() != labels.len() * prod(sample_shape)`.
    pub fn new(features: Vec<f32>, labels: Vec<usize>, sample_shape: &[usize]) -> Self {
        let per: usize = sample_shape.iter().product();
        assert_eq!(features.len(), labels.len() * per, "ClassificationData: features do not match labels x sample_shape");
        ClassificationData { features, labels, sample_shape: sample_shape.to_vec() }
    }

    /// Collate function that keeps the sample shape: inputs are
    /// `[B, ...sample_shape]` (e.g. `[B, C, H, W]` images) on `device`.
    pub fn collator(&self, device: Device) -> impl Fn(Vec<(Vec<f32>, usize)>) -> Batch + Send + Sync + 'static {
        let flat = Self::collate(device);
        let sample_shape = self.sample_shape.clone();
        move |items| {
            let b = flat(items);
            let mut shape = vec![b.targets.numel()];
            shape.extend_from_slice(&sample_shape);
            Batch { inputs: b.inputs.reshape(&shape), targets: b.targets }
        }
    }

    /// Collate `(features, label)` samples into a [`Batch`] on `device`
    /// with flattened inputs `[B, prod(sample_shape)]` (see [`ClassificationData::collator`]).
    pub fn collate(device: Device) -> impl Fn(Vec<(Vec<f32>, usize)>) -> Batch + Send + Sync + 'static {
        move |items: Vec<(Vec<f32>, usize)>| {
            let per = items.first().map_or(0, |(x, _)| x.len());
            let mut xs = Vec::with_capacity(items.len() * per);
            let mut ys = Vec::with_capacity(items.len());
            for (x, y) in items {
                xs.extend_from_slice(&x);
                ys.push(y);
            }
            let n = ys.len();
            let mut shape = vec![n];
            shape.push(per);
            Batch {
                inputs: Tensor::from_vec(xs, &shape, device),
                targets: Tensor::from_indices(&ys, &[n], device),
            }
        }
    }
}

impl Dataset for ClassificationData {
    type Item = (Vec<f32>, usize);
    fn len(&self) -> usize {
        self.labels.len()
    }
    fn get(&self, i: usize) -> (Vec<f32>, usize) {
        let per: usize = self.sample_shape.iter().product();
        (self.features[i * per..(i + 1) * per].to_vec(), self.labels[i])
    }
}

type Collate<I, B> = Arc<dyn Fn(Vec<I>) -> B + Send + Sync>;

/// Batches a [`Dataset`]; see the module docs.
pub struct DataLoader<D: Dataset, B> {
    dataset: Arc<D>,
    collate: Collate<D::Item, B>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
    prefetch: usize,
    seed: u64,
}

impl<D: Dataset + 'static, B: Send + 'static> DataLoader<D, B> {
    /// `collate` turns a `Vec` of samples into one batch value.
    pub fn new(dataset: D, batch_size: usize, collate: impl Fn(Vec<D::Item>) -> B + Send + Sync + 'static) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        DataLoader {
            dataset: Arc::new(dataset),
            collate: Arc::new(collate),
            batch_size,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
            prefetch: 2,
            seed: 0,
        }
    }

    /// Reshuffle every epoch (default: off).
    pub fn shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    /// Skip the last incomplete batch (default: off).
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }

    /// Background threads building batches; 0 builds them on the calling thread.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Batches each worker may prepare ahead (default 2).
    pub fn prefetch(mut self, n: usize) -> Self {
        self.prefetch = n.max(1);
        self
    }

    /// Seed for shuffling (each epoch derives its own order from it).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The wrapped dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Samples per batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Batches per epoch.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// `true` if an epoch yields no batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample indices for `epoch`, in batch order.
    fn order(&self, epoch: u64) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            Rng::new(self.seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)).shuffle(&mut idx);
        }
        idx
    }

    /// Iterate over the batches of `epoch` (determines the shuffle order).
    pub fn epoch(&self, epoch: u64) -> Batches<B> {
        let order = self.order(epoch);
        let batches: Vec<Vec<usize>> = order
            .chunks(self.batch_size)
            .filter(|c| !self.drop_last || c.len() == self.batch_size)
            .map(<[usize]>::to_vec)
            .collect();
        let total = batches.len();
        let (ds, collate) = (self.dataset.clone(), self.collate.clone());
        let make = move |idx: &[usize]| collate(idx.iter().map(|&i| ds.get(i)).collect());
        if self.num_workers == 0 {
            let mut pending = batches;
            pending.reverse();
            return Batches { inner: Inner::Inline { pending, make: Box::new(make) }, next: 0, total };
        }
        let batches = Arc::new(batches);
        let make = Arc::new(make);
        let workers = self.num_workers.min(total.max(1));
        let mut receivers = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let (tx, rx) = sync_channel(self.prefetch);
            let (batches, make) = (batches.clone(), make.clone());
            let handle = std::thread::Builder::new()
                .name(format!("dataloader-{w}"))
                .spawn(move || {
                    for b in (w..batches.len()).step_by(workers) {
                        if tx.send(make(&batches[b])).is_err() {
                            break; // iterator dropped early
                        }
                    }
                })
                .expect("failed to spawn DataLoader worker");
            receivers.push(rx);
            handles.push(handle);
        }
        Batches { inner: Inner::Workers { receivers, handles }, next: 0, total }
    }

    /// Iterate over epoch 0.
    pub fn iter(&self) -> Batches<B> {
        self.epoch(0)
    }
}

impl<D: Dataset + 'static, B: Send + 'static> IntoIterator for &DataLoader<D, B> {
    type Item = B;
    type IntoIter = Batches<B>;
    fn into_iter(self) -> Batches<B> {
        self.iter()
    }
}

type MakeBatch<B> = Box<dyn Fn(&[usize]) -> B + Send>;

enum Inner<B> {
    Inline { pending: Vec<Vec<usize>>, make: MakeBatch<B> },
    Workers { receivers: Vec<Receiver<B>>, handles: Vec<JoinHandle<()>> },
}

/// Iterator over one epoch of batches.
pub struct Batches<B> {
    inner: Inner<B>,
    next: usize,
    total: usize,
}

impl<B> Iterator for Batches<B> {
    type Item = B;

    fn next(&mut self) -> Option<B> {
        if self.next >= self.total {
            return None;
        }
        let i = self.next;
        self.next += 1;
        match &mut self.inner {
            Inner::Inline { pending, make } => pending.pop().map(|idx| make(&idx)),
            Inner::Workers { receivers, handles } => {
                let w = i % receivers.len();
                match receivers[w].recv() {
                    Ok(b) => Some(b),
                    Err(_) => {
                        // the worker exited without sending: it panicked
                        let h = handles.remove(w);
                        match h.join() {
                            Err(payload) => std::panic::resume_unwind(payload),
                            Ok(()) => panic!("DataLoader worker {w} stopped unexpectedly"),
                        }
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.total - self.next;
        (n, Some(n))
    }
}

impl<B> ExactSizeIterator for Batches<B> {}

impl<B> Drop for Batches<B> {
    fn drop(&mut self) {
        if let Inner::Workers { receivers, handles } = &mut self.inner {
            receivers.clear(); // unblocks workers waiting to send
            for h in handles.drain(..) {
                let _ = h.join();
            }
        }
    }
}
