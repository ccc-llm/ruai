//! A character-level GPT trained in the browser on WebGPU, with ruai's nn
//! compiled to wasm32. The page (index.html) calls these functions; the GPU
//! work of a batch of steps is queued without waiting, and only its mean
//! loss is read back (`Tensor::to_vec_async`), so the page stays responsive.

use ruai::nn::{GPTConfig, Module, GPT};
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const TEXT: &str = "\
the quick brown fox jumps over the lazy dog. \
a neural network is a function made of layers. \
each layer multiplies by weights and adds a bias. \
training adjusts the weights to make the loss small. \
the gradient tells us which way the loss goes up, so we step the other way. \
attention lets every token look at the tokens before it. \
convolution slides a small filter over an image. \
rust is fast and safe, and now it can learn too. ";

const STEPS: usize = 600;
const BATCH: usize = 16;

struct Trainer {
    model: GPT,
    cfg: GPTConfig,
    opt: Adam,
    sched: CosineLr,
    vocab: Vec<char>,
    data: Vec<usize>,
    step: usize,
}

thread_local! {
    static TRAINER: RefCell<Option<Trainer>> = const { RefCell::new(None) };
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// Create the WebGPU backend: the adapter's name, or an error.
#[wasm_bindgen]
pub async fn start() -> Result<String, JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_log(&format!("panic: {info}"));
    }));
    if !ruai::backend::webgpu::init().await {
        return Err(JsValue::from_str("this browser has no WebGPU adapter"));
    }
    Ok(ruai::backend::webgpu::adapter_name().unwrap_or_default())
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn web_log(s: &str);
}

/// A new model (seeded): its number of parameters.
#[wasm_bindgen]
pub fn reset(seed: u32) -> usize {
    ruai::rand::manual_seed(seed as u64);
    let text = TEXT.repeat(4);
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();
    let data: Vec<usize> = text.chars().map(|c| vocab.binary_search(&c).unwrap()).collect();
    let cfg = GPTConfig { vocab_size: vocab.len(), max_len: 64, dim: 64, n_heads: 4, n_layers: 2, dropout: 0.0 };
    let model = GPT::new(cfg, Device::WebGpu);
    let n = model.num_parameters();
    let base_lr = 3e-3;
    let opt = Adam::adamw(model.parameters(), base_lr, 0.01);
    let sched = CosineLr { base_lr, min_lr: base_lr * 0.1, warmup: 30, total: STEPS };
    TRAINER.with(|t| *t.borrow_mut() = Some(Trainer { model, cfg, opt, sched, vocab, data, step: 0 }));
    n
}

/// The total number of training steps.
#[wasm_bindgen]
pub fn total_steps() -> usize {
    STEPS
}

/// Run `n` training steps: [step reached, mean loss, milliseconds].
#[wasm_bindgen]
pub async fn train(n: usize) -> Vec<f64> {
    let t0 = now_ms();
    // queue the work (no reads), keeping the losses on the GPU
    let (sum, steps, step) = TRAINER.with(|t| {
        let mut t = t.borrow_mut();
        let t = t.as_mut().expect("reset first");
        let block = t.cfg.max_len;
        let mut sum: Option<Tensor> = None;
        let mut k = 0;
        while k < n && t.step < STEPS {
            t.sched.apply(&mut t.opt, t.step);
            t.opt.zero_grad();
            let mut xs = Vec::with_capacity(BATCH * block);
            let mut ys = Vec::with_capacity(BATCH * block);
            for _ in 0..BATCH {
                let s = ruai::rand::randint(t.data.len() - block - 1);
                xs.extend_from_slice(&t.data[s..s + block]);
                ys.extend_from_slice(&t.data[s + 1..s + block + 1]);
            }
            let logits = t.model.forward_ids(&xs, BATCH, block).reshape(&[BATCH * block, t.cfg.vocab_size]);
            let loss = F::cross_entropy(&logits, &ys, 0.0);
            loss.backward();
            t.opt.step();
            let l = loss.detach();
            sum = Some(match sum {
                Some(s) => &s + &l,
                None => l,
            });
            t.step += 1;
            k += 1;
        }
        (sum, k, t.step)
    });
    let loss = match sum {
        Some(s) => s.to_vec_async().await[0] as f64 / steps.max(1) as f64,
        None => f64::NAN,
    };
    vec![step as f64, loss, now_ms() - t0]
}

/// Continue `prompt` by `n` characters (greedy, temperature 0.8 sampling when `sample`).
#[wasm_bindgen]
pub async fn generate(prompt: String, n: usize, sample: bool) -> String {
    let mut out = prompt.clone();
    for _ in 0..n {
        let next = TRAINER.with(|t| {
            let t = t.borrow();
            let t = t.as_ref().expect("reset first");
            let ids: Vec<usize> = out.chars().filter_map(|c| t.vocab.binary_search(&c).ok()).collect();
            let ids = &ids[ids.len().saturating_sub(t.cfg.max_len)..];
            let logits = ruai::no_grad(|| t.model.forward_ids(ids, 1, ids.len()));
            logits.narrow(1, ids.len() - 1, 1).reshape(&[t.cfg.vocab_size])
        });
        let logits = next.to_vec_async().await;
        let c = TRAINER.with(|t| {
            let t = t.borrow();
            let t = t.as_ref().unwrap();
            let i = if sample {
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p: Vec<f32> = logits.iter().map(|l| ((l - m) / 0.8).exp()).collect();
                let s: f32 = p.iter().sum();
                ruai::rand::sample_categorical(&p.iter().map(|x| x / s).collect::<Vec<_>>())
            } else {
                (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap()
            };
            t.vocab[i]
        });
        out.push(c);
    }
    out
}

/// matmul n×n on WebGPU: GFLOP/s over `iters` products.
#[wasm_bindgen]
pub async fn bench_matmul(n: usize, iters: usize) -> f64 {
    let a = Tensor::randn(&[n, n], Device::WebGpu);
    let b = Tensor::randn(&[n, n], Device::WebGpu);
    // warm up (shader compilation), then time
    let _ = a.matmul(&b).sum_all().to_vec_async().await;
    let t0 = now_ms();
    let mut acc = a.matmul(&b).sum_all();
    for _ in 1..iters {
        acc = &acc + &a.matmul(&b).sum_all();
    }
    let _ = acc.to_vec_async().await;
    let ms = now_ms() - t0;
    2.0 * (n as f64).powi(3) * iters as f64 / ms / 1e6
}

/// The loss curve as JSON for `ruaiDraw` (ruai::plot's canvas renderer).
#[wasm_bindgen]
pub fn loss_plot(steps: Vec<f64>, losses: Vec<f64>, width: f64, height: f64) -> String {
    use ruai::plot::{Figure, Style};
    let mut fig = Figure::new(width / 100.0, height / 100.0);
    let ax = fig.add_subplot(1, 1, 1);
    let a = &mut fig.axes[ax];
    a.plot(steps, losses, &Style::default().label("training loss"));
    a.title = "GPT on WebGPU".into();
    a.xlabel = "step".into();
    a.ylabel = "cross-entropy".into();
    a.grid_lines = true;
    ruai::plot::render::canvas_json(&fig)
}

/// The script that draws `loss_plot`'s JSON on a canvas.
#[wasm_bindgen]
pub fn canvas_js() -> String {
    ruai::plot::render::CANVAS_JS.to_string()
}
