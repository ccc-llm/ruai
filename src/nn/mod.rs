//! Neural-network layers (like `torch.nn`).

pub mod gpt2;
/// Basic layers: linear, convolution, pooling, normalization, embeddings, activations, MLPs.
pub mod layers;
pub mod llama;
/// The [`Module`] trait and [`Sequential`].
pub mod module;
pub mod resnet;
pub mod seq2seq;
/// Attention, transformer blocks, the KV cache and a small GPT.
pub mod transformer;
pub mod vit;

pub use layers::*;
pub use module::{prefixed, Module, Sequential};
pub use transformer::{GPTConfig, KvCache, LayerKv, MultiHeadAttention, TransformerBlock, GPT};
