//! # ruai
//!
//! A Swiss-army toolkit for AI in Rust, built as one crate whose parts are
//! cargo features (`nn`, `world`, `py` and `agent` are on by default):
//!
//! * [`nn`] (feature `nn`) — a PyTorch-like deep learning engine: tensors on
//!   CPU / Apple GPU (`mps`) / NVIDIA GPU (`cuda`), autograd with second
//!   derivatives, layers, optimizers, GPT-2 / Llama / ResNet / ViT,
//!   quantization, safetensors. Its items are also re-exported here, so
//!   `ruai::Tensor` is `ruai::nn::Tensor`.
//! * [`world`] (feature `world`) — reinforcement-learning environments with
//!   the Gymnasium API (CartPole, MountainCar, Pendulum, Acrobot,
//!   FrozenLake), matching Gymnasium bit for bit for the same seed.
//! * [`py`] (feature `py`) — py0, a typed subset of Python translated to
//!   readable Rust: `ruai py run prog.py`. Translated programs depend on
//!   `ruai` with only the [`pyrt`] runtime features: `pyrt` itself, and
//!   `pyrt-re`, `pyrt-http`, `pyrt-numpy`, `pyrt-sklearn`, `pyrt-gymnasium`,
//!   `pyrt-torch` for the packages of the same names (Python's standard
//!   library and subsets of NumPy, scikit-learn, Gymnasium and PyTorch).
//! * feature `agent` — `ruai agent`, teams of LLM agents, written in py0.
//! * [`rng`] — NumPy-compatible random numbers (`SeedSequence`, `PCG64`) and
//!   the MT19937 of NumPy's `RandomState` and PyTorch.
//!
//! The `ruai` command line tool dispatches to the parts like `git`:
//! `ruai nn generate ...`, `ruai py run ...`, `ruai world run ...`,
//! `ruai agent chat`; an unknown `ruai NAME` runs `ruai-NAME` from `PATH`.
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
//! re-exports), [`world`] and [`rng`], for the `ruai` command line, and for
//! the py0 language and the behavior of its packages. [`py`] and [`pyrt`] are
//! public because translated programs use them; their Rust API follows the
//! translator and may change in minor versions.

#![warn(missing_docs)]

#[cfg(feature = "nn")]
pub mod nn;
#[cfg(feature = "py")]
#[allow(missing_docs)]
pub mod py;
#[cfg(feature = "pyrt")]
#[allow(missing_docs)]
pub mod pyrt;
pub mod rng;
#[cfg(feature = "world")]
pub mod world;
#[cfg(feature = "nn")]
pub use nn::*;
