//! LoRA adapters and gradient checkpointing.
//! Set `RUSTNN_DEVICE=mps` to run on the GPU.

use ruai::functional as F;
use ruai::nn::gpt2::{GPT2Config, GPT2};
use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::nn::{self, Linear, Module};
use ruai::optim::{Adam, Optimizer};
use ruai::{no_grad, rand, serialize, Device, Tensor};

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

fn tiny_llama() -> Llama {
    Llama::new(LlamaConfig::new(23, 32, 2, 4, 2, 16), dev())
}

fn lm_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let v = logits.dim(-1);
    F::cross_entropy(&logits.reshape(&[targets.len(), v]), targets, 0.0)
}

#[test]
fn lora_linear_starts_as_identity_and_merges() {
    let _l = lock();
    rand::manual_seed(0);
    let mut lin = Linear::new(8, 6, dev());
    let x = Tensor::randn(&[4, 8], dev());
    let before = no_grad(|| lin.forward(&x)).to_vec();
    lin.add_lora(2, 4.0);
    assert_eq!(no_grad(|| lin.forward(&x)).to_vec(), before, "B = 0: no change at init");
    let names: Vec<String> = lin.trainable_parameters().iter().map(|t| format!("{:?}", t.shape())).collect();
    assert_eq!(names, ["[8, 2]", "[2, 6]"], "only the adapter trains");
    let w0 = lin.weight.to_vec();
    let mut opt = Adam::new(lin.trainable_parameters(), 0.05);
    let target = Tensor::randn(&[4, 6], dev());
    for _ in 0..20 {
        opt.zero_grad();
        F::mse_loss(&lin.forward(&x), &target).backward();
        opt.step();
    }
    assert_eq!(lin.weight.to_vec(), w0, "frozen weight is untouched");
    let adapted = no_grad(|| lin.forward(&x)).to_vec();
    assert!(max_abs_diff(&adapted, &before) > 1e-3);
    lin.merge_lora();
    assert!(lin.lora.is_none());
    assert!(max_abs_diff(&no_grad(|| lin.forward(&x)).to_vec(), &adapted) < 1e-5);
}

#[test]
fn llama_lora_finetuning() {
    let _l = lock();
    rand::manual_seed(1);
    let mut m = tiny_llama();
    let n = m.add_lora(4, 8.0, &["q_proj", "v_proj"]);
    assert_eq!(n, 4, "2 layers x (q, v)");
    let trainable = m.trainable_parameters();
    assert_eq!(trainable.len(), 8);
    let lora = nn::lora_state_dict(&m);
    assert!(lora.iter().all(|(n, _)| n.contains("_proj.lora_")), "{:?}", lora.iter().map(|x| &x.0).collect::<Vec<_>>());
    let emb = m.embed_tokens.weight.to_vec();

    let seq = [1usize, 2, 3, 4, 5, 6, 7, 8, 9];
    let inp = Tensor::from_indices(&seq[..8], &[1, 8], dev());
    let mut opt = Adam::new(trainable, 0.02);
    let first = lm_loss(&m.forward_tokens(&inp), &seq[1..]).item();
    for _ in 0..60 {
        opt.zero_grad();
        lm_loss(&m.forward_tokens(&inp), &seq[1..]).backward();
        opt.step();
    }
    let last = no_grad(|| lm_loss(&m.forward_tokens(&inp), &seq[1..]).item());
    assert!(last < 0.5 * first, "loss {first} -> {last}");
    assert_eq!(m.embed_tokens.weight.to_vec(), emb, "frozen parameters are untouched");

    // adapters saved alone load into a fresh copy of the base model
    let logits = no_grad(|| m.forward_tokens(&inp)).to_vec();
    let path = std::env::temp_dir().join(format!("ruai_lora_{}.safetensors", std::process::id()));
    serialize::save_tensors(&path, &nn::lora_state_dict(&m)).unwrap();
    let base = serialize::to_safetensors_bytes(&m.state_dict().into_iter().filter(|(n, _)| !n.contains(".lora_")).collect::<Vec<_>>(), None).unwrap();
    let mut fresh = tiny_llama();
    fresh.add_lora(4, 8.0, &["q_proj", "v_proj"]);
    serialize::load_state_dict(&fresh, &serialize::from_safetensors_bytes(&base).unwrap().0, false).unwrap();
    let report = serialize::load_state_dict(&fresh, &serialize::load_tensors(&path).unwrap(), false).unwrap();
    assert!(report.unexpected.is_empty());
    std::fs::remove_file(&path).ok();
    assert!(max_abs_diff(&no_grad(|| fresh.forward_tokens(&inp)).to_vec(), &logits) < 1e-5);

    // merged model gives the same logits and exports in HF layout
    m.merge_lora();
    assert!(max_abs_diff(&no_grad(|| m.forward_tokens(&inp)).to_vec(), &logits) < 1e-4);
    assert!(nn::lora_state_dict(&m).is_empty());
    assert!(!m.hf_state_dict().is_empty());
}

#[test]
#[should_panic(expected = "merge the LoRA adapters")]
fn hf_export_refuses_unmerged_lora() {
    let mut m = tiny_llama();
    m.add_lora(2, 2.0, &["mlp"]);
    m.hf_state_dict();
}

#[test]
fn lora_on_int8_base_trains() {
    let _l = lock();
    rand::manual_seed(2);
    let mut m = tiny_llama();
    m.quantize_int8();
    assert_eq!(m.add_lora(4, 8.0, &["self_attn", "mlp"]), 14);
    let seq = [3usize, 1, 4, 1, 5, 9, 2, 6, 5];
    let inp = Tensor::from_indices(&seq[..8], &[1, 8], dev());
    let mut opt = Adam::new(m.trainable_parameters(), 0.02);
    let first = no_grad(|| lm_loss(&m.forward_tokens(&inp), &seq[1..]).item());
    for _ in 0..60 {
        opt.zero_grad();
        lm_loss(&m.forward_tokens(&inp), &seq[1..]).backward();
        opt.step();
    }
    let last = no_grad(|| lm_loss(&m.forward_tokens(&inp), &seq[1..]).item());
    assert!(last < 0.5 * first, "loss {first} -> {last}");
}

/// Gradients of every trainable parameter after one backward pass.
fn grads(m: &dyn Module, loss: impl Fn() -> Tensor) -> Vec<Vec<f32>> {
    m.zero_grad();
    loss().backward();
    m.trainable_parameters().iter().map(|p| p.grad().map_or_else(Vec::new, |g| g.to_vec())).collect()
}

fn assert_same_grads(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(!x.is_empty(), "{what}: parameter {i} got no gradient");
        assert!(max_abs_diff(x, y) < 1e-5, "{what}: parameter {i} differs by {}", max_abs_diff(x, y));
    }
}

#[test]
fn gradient_checkpointing_gives_the_same_gradients() {
    let _l = lock();
    rand::manual_seed(3);
    let seq = [1usize, 2, 3, 4, 5, 6, 7, 8, 9];

    let llama = tiny_llama();
    let inp = Tensor::from_indices(&seq[..8], &[1, 8], dev());
    let loss = || lm_loss(&llama.forward_tokens(&inp), &seq[1..]);
    let plain = grads(&llama, loss);
    llama.set_gradient_checkpointing(true);
    assert_same_grads(&plain, &grads(&llama, loss), "llama");

    // with LoRA the block inputs do not require grad, only the adapters inside do
    let mut lora = tiny_llama();
    lora.add_lora(2, 4.0, &["q_proj"]);
    let loss = || lm_loss(&lora.forward_tokens(&inp), &seq[1..]);
    let plain = grads(&lora, loss);
    lora.set_gradient_checkpointing(true);
    assert_same_grads(&plain, &grads(&lora, loss), "llama + lora");

    // dropout: the recomputation must replay the same masks
    let gpt = nn::GPT::new(nn::GPTConfig { vocab_size: 23, max_len: 16, dim: 32, n_heads: 4, n_layers: 2, dropout: 0.2 }, dev());
    let loss = || lm_loss(&gpt.forward_tokens(&inp), &seq[1..]);
    rand::manual_seed(10);
    let plain = grads(&gpt, loss);
    gpt.set_gradient_checkpointing(true);
    rand::manual_seed(10);
    assert_same_grads(&plain, &grads(&gpt, loss), "gpt with dropout");

    let g2 = GPT2::new(GPT2Config { vocab_size: 23, n_positions: 16, n_embd: 32, n_layer: 2, n_head: 4 }, dev());
    let loss = || lm_loss(&g2.forward_tokens(&inp), &seq[1..]);
    let plain = grads(&g2, loss);
    g2.set_gradient_checkpointing(true);
    assert_same_grads(&plain, &grads(&g2, loss), "gpt2");
}

#[test]
fn grad_checkpoint_function() {
    let _l = lock();
    rand::manual_seed(4);
    let w = Tensor::randn(&[3, 3], dev()).requires_grad_();
    let x = Tensor::randn(&[2, 3], dev()).requires_grad_();
    let f = |w: &Tensor, x: &Tensor| (x.matmul(w).tanh() * 2.0).sum_all();
    f(&w, &x).backward();
    let (gw, gx) = (w.grad().unwrap().to_vec(), x.grad().unwrap().to_vec());
    w.zero_grad();
    x.zero_grad();
    let wc = w.clone();
    let y = F::grad_checkpoint(&[&x], vec![w.clone()], move |xs| (xs[0].matmul(&wc).tanh() * 2.0).sum_all());
    assert_eq!(y.op_name(), Some("grad_checkpoint"));
    y.backward();
    assert!(max_abs_diff(&w.grad().unwrap().to_vec(), &gw) < 1e-6);
    assert!(max_abs_diff(&x.grad().unwrap().to_vec(), &gx) < 1e-6);
    // under no_grad it is just a call
    let z = no_grad(|| F::grad_checkpoint(&[&x], vec![w.clone()], |xs| xs[0].sum_all()));
    assert!(z.op_name().is_none());
}

#[test]
fn gpt2_lora() {
    let _l = lock();
    rand::manual_seed(5);
    let mut g = GPT2::new(GPT2Config { vocab_size: 23, n_positions: 16, n_embd: 32, n_layer: 2, n_head: 4 }, dev());
    assert_eq!(g.add_lora(2, 4.0, &["c_attn"]), 2);
    assert_eq!(g.add_lora(2, 4.0, &["c_proj"]), 4, "attn.c_proj and mlp.c_proj");
    assert_eq!(g.trainable_parameters().len(), 12);
}
