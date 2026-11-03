//! # ruai
//!
//! A Swiss-army toolkit for AI in Rust, built as one crate whose parts are
//! cargo features (`nn`, `world` and `plot` are on by default):
//!
//! * [`nn`] (feature `nn`) — a PyTorch-like deep learning engine: tensors on
//!   CPU / Apple GPU (`mps`) / NVIDIA GPU (`cuda`) / any GPU and browsers
//!   through WebGPU (`webgpu`), autograd with second
//!   derivatives, layers, optimizers, GPT-2 / Llama / ResNet / ViT,
//!   quantization, safetensors. Its items are also re-exported here, so
//!   `ruai::Tensor` is `ruai::nn::Tensor`.
//! * [`world`] (feature `world`) — reinforcement-learning environments with
//!   the Gymnasium API (CartPole, MountainCar, Pendulum, Acrobot,
//!   FrozenLake), matching Gymnasium bit for bit for the same seed.
//! * [`plot`] (feature `plot`) — plots in matplotlib's style, drawn on an
//!   HTML canvas or as SVG.
//! * [`rng`] — NumPy-compatible random numbers (`SeedSequence`, `PCG64`) and
//!   the MT19937 of NumPy's `RandomState` and PyTorch.
//!
//! The `ruai` command line tool dispatches to the parts like `git`:
//! `ruai nn generate ...`, `ruai world run ...`; an unknown `ruai NAME` runs
//! `ruai-NAME` from `PATH`. Typed Python compiled to Rust (formerly `ruai py`,
//! py0) and `ruai agent` are now [rupy](https://crates.io/crates/rupy), whose
//! `ruai-py` and `ruai-agent` keep those commands working.
//!
//! ```
//! use ruai::{Tensor, Device};
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).requires_grad_();
//! let y = (&x * &x).sum_all();
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
//! ```
//!
//! ## Stability
//!
//! ruai follows semantic versioning for the Rust API of [`nn`] (and its
//! re-exports), [`world`], [`plot`] and [`rng`], and for the `ruai` command line.

#![warn(missing_docs)]

#[cfg(feature = "nn")]
pub mod nn;
pub mod rng;
#[cfg(feature = "plot")]
pub mod plot;
#[cfg(feature = "world")]
pub mod world;
#[cfg(feature = "nn")]
pub use nn::*;
