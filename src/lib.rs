//! # rust-nn
//!
//! A small PyTorch-like deep learning engine:
//!
//! * [`Tensor`] — n-dimensional `f32` arrays with reverse-mode autograd
//! * [`nn`] — layers (Linear, Conv2d, BatchNorm2d, LayerNorm, Embedding,
//!   MultiHeadAttention, TransformerBlock, GPT, ...)
//! * [`functional`] — softmax, cross-entropy, conv2d, pooling, attention, ...
//! * [`optim`] — SGD, Adam, AdamW, gradient clipping, LR schedules
//! * [`serialize`] — save / load model weights
//! * [`backend`] — the device abstraction (CPU today; CUDA and MPS plug in
//!   by implementing [`backend::Backend`])
//!
//! ```
//! use rustnn::{Tensor, Device};
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).requires_grad_();
//! let y = (&x * &x).sum_all();
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
//! ```

pub mod backend;
pub mod device;
pub mod functional;
pub mod nn;
mod ops;
pub mod optim;
pub mod rand;
pub mod serialize;
pub mod shape;
pub mod tensor;

pub use device::Device;
pub use tensor::{is_grad_enabled, no_grad, NoGradGuard, Tensor};

/// Commonly used items: `use rustnn::prelude::*;`
pub mod prelude {
    pub use crate::functional as F;
    pub use crate::nn::{self, Module};
    pub use crate::optim::{self, Optimizer};
    pub use crate::{no_grad, Device, Tensor};
}
