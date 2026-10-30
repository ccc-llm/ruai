//! # ruai
//!
//! A small PyTorch-like deep learning engine:
//!
//! * [`Tensor`] — n-dimensional arrays (F32 / F16 / BF16 / I32 / I8) on CPU,
//!   Apple GPU (`mps` feature) or NVIDIA GPU (`cuda` feature), with
//!   reverse-mode autograd
//! * [`nn`] — layers (Linear, Conv2d, BatchNorm2d, LayerNorm, RMSNorm,
//!   Embedding, MultiHeadAttention, TransformerBlock, SwiGLU, GPT, ...)
//! * [`functional`] — softmax, cross-entropy, conv2d, pooling, attention, ...
//! * [`optim`] — SGD, Adam, AdamW, gradient clipping, LR schedulers ([`optim::lr`])
//! * [`serialize`] — save / load weights (safetensors, PyTorch/HF compatible)
//! * [`amp`] — mixed precision: `autocast` (F16/BF16) and `GradScaler`
//! * [`checkpoint`] — resumable training checkpoints (model + optimizer + RNG)
//! * [`nn::gpt2`] / [`tokenizer`] / [`generation`] — pretrained GPT-2, BPE, sampling
//! * [`nn::llama`] — Llama-family decoder (RMSNorm, RoPE, SwiGLU, GQA), Hugging Face compatible
//! * [`nn::resnet`] / [`nn::vit`] / [`nn::seq2seq`] — ResNet, Vision Transformer, encoder-decoder Transformer
//! * [`data`] — `Dataset` / `DataLoader` with multi-threaded prefetching
//! * [`metrics`] — running averages, accuracy, throughput
//! * [`backend`] — the device abstraction: every op is built from the
//!   primitives of [`backend::Backend`] (CPU, MPS and CUDA implement them)
//!
//! ```
//! use ruai::{Tensor, Device};
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).requires_grad_();
//! let y = (&x * &x).sum_all();
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
//! ```
//!
//! A training step looks like PyTorch:
//!
//! ```
//! use ruai::prelude::*;
//! use ruai::optim::Adam;
//!
//! let model = nn::Sequential::new()
//!     .add(nn::Linear::new(4, 16, Device::Cpu))
//!     .add(nn::ReLU)
//!     .add(nn::Linear::new(16, 3, Device::Cpu));
//! let mut opt = Adam::new(model.parameters(), 1e-2);
//! let x = Tensor::randn(&[8, 4], Device::Cpu);
//! let y = [0usize, 1, 2, 0, 1, 2, 0, 1];
//! for _ in 0..5 {
//!     opt.zero_grad();
//!     let loss = F::cross_entropy(&model.forward(&x), &y, 0.0);
//!     loss.backward();
//!     opt.step();
//! }
//! ```

#![warn(missing_docs)]

pub mod amp;
pub mod backend;
pub mod checkpoint;
pub mod data;
pub mod device;
pub mod functional;
pub mod generation;
pub mod metrics;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod rand;
pub mod serialize;
pub mod shape;
pub mod tensor;
pub mod tokenizer;

pub use backend::DType;
pub use device::Device;
pub use ops::IntoIndices;
pub use tensor::{autocast, autocast_dtype, is_grad_enabled, no_grad, NoGradGuard, Tensor};

/// Commonly used items: `use ruai::prelude::*;`
pub mod prelude {
    pub use crate::functional as F;
    pub use crate::nn::{self, Module};
    pub use crate::data::{DataLoader, Dataset};
    pub use crate::generation::Sampler;
    pub use crate::optim::{self, lr::LrScheduler, Optimizer};
    pub use crate::{autocast, no_grad, DType, Device, IntoIndices, Tensor};
}
