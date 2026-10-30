//! # ruai
//!
//! A Swiss-army toolkit for AI in Rust, built as one crate whose parts are
//! cargo features (all on by default):
//!
//! * [`nn`] (feature `nn`) — a PyTorch-like deep learning engine: tensors on
//!   CPU / Apple GPU / NVIDIA GPU, autograd, layers, optimizers, GPT-2 /
//!   Llama / ResNet / ViT, quantization, safetensors. Its items are also
//!   re-exported here, so `ruai::Tensor` is `ruai::nn::Tensor`.
//! * [`world`] (feature `world`) — reinforcement-learning environments with
//!   the Gymnasium API (CartPole, MountainCar, Pendulum, Acrobot,
//!   FrozenLake), matching Gymnasium bit for bit for the same seed.
//! * [`rng`] — NumPy-compatible random numbers (`SeedSequence`, `PCG64`).
//!
//! The `ruai` command line tool dispatches to the parts like `git`:
//! `ruai nn generate ...`; an unknown `ruai NAME` runs `ruai-NAME` from `PATH`.
//!
//! ```
//! use ruai::{Tensor, Device};
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).requires_grad_();
//! let y = (&x * &x).sum_all();
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
//! ```

#![warn(missing_docs)]

#[cfg(feature = "nn")]
pub mod nn;
pub mod rng;
#[cfg(feature = "world")]
pub mod world;
#[cfg(feature = "nn")]
pub use nn::*;
