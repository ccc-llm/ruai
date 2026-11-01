//! MNIST with a small CNN, in the manner of ConvNetJS's MNIST demo: conv 5×5×8
//! → ReLU → max pool 2 → conv 5×5×16 → ReLU → max pool 2 → linear → softmax,
//! trained with Adam on batches of 20 randomly shifted digits. The page shows
//! the activations of every layer and the predictions on the test digits.

use crate::now_ms;
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const BATCH: usize = 20;
const PIX: usize = 28 * 28;

struct Net {
    c1: nn::Conv2d,
    c2: nn::Conv2d,
    fc: nn::Linear,
    pool: nn::MaxPool2d,
}

impl Net {
    fn new() -> Net {
        let d = Device::Cpu;
        Net { c1: nn::Conv2d::new(1, 8, 5, 1, 2, d), c2: nn::Conv2d::new(8, 16, 5, 1, 2, d), fc: nn::Linear::new(16 * 7 * 7, 10, d), pool: nn::MaxPool2d::new(2, 2) }
    }

    fn parameters(&self) -> Vec<Tensor> {
        [self.c1.parameters(), self.c2.parameters(), self.fc.parameters()].concat()
    }

    /// The logits of [n, 1, 28, 28] images, and the activations of each layer.
    fn forward(&self, x: &Tensor) -> (Tensor, [Tensor; 4]) {
        let n = x.shape()[0];
        let a1 = self.c1.forward(x).relu();
        let p1 = self.pool.forward(&a1);
        let a2 = self.c2.forward(&p1).relu();
        let p2 = self.pool.forward(&a2);
        let logits = self.fc.forward(&p2.reshape(&[n, 16 * 7 * 7]));
        (logits, [a1, p1, a2, p2])
    }
}

struct Mnist {
    data: Vec<u8>,
    train: usize,
    test: usize,
    net: Net,
    opt: Adam,
    rng: ruai::rng::Generator,
    seen: usize,
}

thread_local! {
    static MNIST: RefCell<Option<Mnist>> = const { RefCell::new(None) };
}

impl Mnist {
    fn image(&self, test: bool, i: usize) -> &[u8] {
        let at = 12 + (if test { self.train + i } else { i }) * PIX;
        &self.data[at..at + PIX]
    }

    fn label(&self, test: bool, i: usize) -> usize {
        self.data[12 + (self.train + self.test) * PIX + if test { self.train + i } else { i }] as usize
    }

    /// Image `i` as floats in [0, 1], shifted by (dx, dy) pixels (zeros come in).
    fn pixels(&self, test: bool, i: usize, dx: i32, dy: i32, out: &mut Vec<f32>) {
        let img = self.image(test, i);
        for y in 0..28i32 {
            for x in 0..28i32 {
                let (sx, sy) = (x - dx, y - dy);
                let v = if (0..28).contains(&sx) && (0..28).contains(&sy) { img[(sy * 28 + sx) as usize] } else { 0 };
                out.push(v as f32 / 255.0);
            }
        }
    }

    fn batch(&self, test: bool, ids: &[usize]) -> Tensor {
        let mut xs = Vec::with_capacity(ids.len() * PIX);
        for &i in ids {
            self.pixels(test, i, 0, 0, &mut xs);
        }
        Tensor::from_vec(xs, &[ids.len(), 1, 28, 28], Device::Cpu)
    }
}

fn softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = l.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Take the data file (see site/tools/mnist.mjs): [train count, test count].
#[wasm_bindgen]
pub fn mnist_load(bytes: Vec<u8>) -> Result<Vec<u32>, JsValue> {
    if bytes.len() < 12 || &bytes[..4] != b"MNST" {
        return Err(JsValue::from_str("not an MNIST data file"));
    }
    let n = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (train, test) = (n(4), n(8));
    if bytes.len() != 12 + (train + test) * (PIX + 1) {
        return Err(JsValue::from_str("the MNIST data file is truncated"));
    }
    ruai::rand::manual_seed(1);
    let net = Net::new();
    let opt = Adam::new(net.parameters(), 0.002);
    let m = Mnist { data: bytes, train, test, net, opt, rng: ruai::rng::Generator::new(1), seen: 0 };
    MNIST.with(|s| *s.borrow_mut() = Some(m));
    Ok(vec![train as u32, test as u32])
}

/// A new, untrained network.
#[wasm_bindgen]
pub fn mnist_reset(seed: u32) {
    MNIST.with(|s| {
        let mut s = s.borrow_mut();
        let m = s.as_mut().expect("mnist_load first");
        ruai::rand::manual_seed(seed as u64);
        m.net = Net::new();
        m.opt = Adam::new(m.net.parameters(), 0.002);
        m.rng = ruai::rng::Generator::new(seed as u64);
        m.seen = 0;
    })
}

/// `steps` batches of 20 shifted training digits: [examples seen, mean loss,
/// training accuracy of these batches, milliseconds per example].
#[wasm_bindgen]
pub fn mnist_train(steps: usize) -> Vec<f64> {
    MNIST.with(|s| {
        let mut s = s.borrow_mut();
        let m = s.as_mut().expect("mnist_load first");
        let t0 = now_ms();
        let (mut loss, mut correct) = (0.0, 0);
        for _ in 0..steps {
            let mut xs = Vec::with_capacity(BATCH * PIX);
            let mut ys = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                let i = m.rng.integers(0, m.train as i64) as usize;
                let (dx, dy) = (m.rng.integers(-2, 3) as i32, m.rng.integers(-2, 3) as i32);
                m.pixels(false, i, dx, dy, &mut xs);
                ys.push(m.label(false, i));
            }
            m.opt.zero_grad();
            let (logits, _) = m.net.forward(&Tensor::from_vec(xs, &[BATCH, 1, 28, 28], Device::Cpu));
            let l = F::cross_entropy(&logits, &ys, 0.0);
            l.backward();
            m.opt.step();
            loss += l.item() as f64;
            let v = logits.detach().to_vec();
            correct += ys.iter().enumerate().filter(|&(k, &y)| argmax(&v[10 * k..10 * k + 10]) == y).count();
            m.seen += BATCH;
        }
        let n = (steps * BATCH).max(1) as f64;
        vec![m.seen as f64, loss / steps.max(1) as f64, correct as f64 / n, (now_ms() - t0) / n]
    })
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

/// Test digits `start .. start + count`: per digit its label, then the three
/// likeliest classes and their probabilities (7 numbers each).
#[wasm_bindgen]
pub fn mnist_test(start: usize, count: usize) -> Vec<f32> {
    MNIST.with(|s| {
        let s = s.borrow();
        let m = s.as_ref().expect("mnist_load first");
        let end = (start + count).min(m.test);
        let ids: Vec<usize> = (start..end).collect();
        let logits = no_grad(|| m.net.forward(&m.batch(true, &ids)).0).to_vec();
        let mut out = Vec::with_capacity(ids.len() * 7);
        for (k, &i) in ids.iter().enumerate() {
            let p = softmax(&logits[10 * k..10 * k + 10]);
            let mut order: Vec<usize> = (0..10).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            out.push(m.label(true, i) as f32);
            for &c in &order[..3] {
                out.extend_from_slice(&[c as f32, p[c]]);
            }
        }
        out
    })
}

/// Test digit `i` as bytes (for drawing).
#[wasm_bindgen]
pub fn mnist_image(test: bool, i: usize) -> Vec<u8> {
    MNIST.with(|s| s.borrow().as_ref().map(|m| m.image(test, i).to_vec()).unwrap_or_default())
}

/// The activations for 784 pixels in [0, 1]: conv1 (8×28×28), pool1 (8×14×14),
/// conv2 (16×14×14), pool2 (16×7×7) and the 10 probabilities, one after another.
#[wasm_bindgen]
pub fn mnist_layers(pixels: Vec<f32>) -> Vec<f32> {
    MNIST.with(|s| {
        let s = s.borrow();
        let m = s.as_ref().expect("mnist_load first");
        let (logits, acts) = no_grad(|| m.net.forward(&Tensor::from_vec(pixels, &[1, 1, 28, 28], Device::Cpu)));
        let mut out = Vec::new();
        for a in &acts {
            out.extend(a.to_vec());
        }
        out.extend(softmax(&logits.to_vec()));
        out
    })
}

/// The first layer's 8 filters (5×5 each).
#[wasm_bindgen]
pub fn mnist_filters() -> Vec<f32> {
    MNIST.with(|s| s.borrow().as_ref().map(|m| m.net.c1.weight.to_vec()).unwrap_or_default())
}
