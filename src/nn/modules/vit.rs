//! Vision Transformer (Dosovitskiy et al. 2021): the image is cut into
//! patches (a strided convolution), a learned `[CLS]` token and position
//! embeddings are added, and a stack of pre-LN transformer encoder blocks
//! is applied; the classifier reads the final `[CLS]` vector.

use super::layers::{Conv2d, LayerNorm, Linear};
use super::module::{prefixed, Module};
use super::transformer::TransformerBlock;
use crate::nn::device::Device;
use crate::nn::functional as F;
use crate::nn::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Vision Transformer hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViTConfig {
    /// Input height and width (square images).
    pub image_size: usize,
    /// Patch height and width.
    pub patch_size: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Output classes.
    pub num_classes: usize,
    /// Model width.
    pub dim: usize,
    /// Number of encoder blocks.
    pub depth: usize,
    /// Attention heads.
    pub heads: usize,
    /// MLP hidden size.
    pub mlp_dim: usize,
    /// Dropout probability (training only).
    pub dropout: f32,
}

impl ViTConfig {
    /// ViT-B/16 at 224x224 (86.6M parameters with 1000 classes).
    pub fn base_16(num_classes: usize) -> Self {
        ViTConfig {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            num_classes,
            dim: 768,
            depth: 12,
            heads: 12,
            mlp_dim: 3072,
            dropout: 0.0,
        }
    }

    /// A small ViT for low-resolution images (`mlp_dim = 2 * dim`).
    pub fn tiny(image_size: usize, patch_size: usize, in_channels: usize, num_classes: usize, dim: usize, depth: usize, heads: usize) -> Self {
        ViTConfig { image_size, patch_size, in_channels, num_classes, dim, depth, heads, mlp_dim: 2 * dim, dropout: 0.0 }
    }

    /// Number of patches per image.
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

/// Vision Transformer classifier: `[N, C, H, W]` -> logits `[N, classes]`.
pub struct ViT {
    /// Hyperparameters.
    pub config: ViTConfig,
    /// Patch embedding: a `patch_size`-strided convolution.
    pub patch_embed: Conv2d,
    /// `[1, 1, dim]`
    pub cls_token: Tensor,
    /// `[1, num_patches + 1, dim]`
    pub pos_embed: Tensor,
    /// Encoder blocks (bidirectional).
    pub blocks: Vec<TransformerBlock>,
    /// Final layer norm.
    pub norm: LayerNorm,
    /// Classifier on the `[CLS]` vector.
    pub head: Linear,
    training: AtomicBool,
}

impl ViT {
    /// Randomly initialized model.
    pub fn new(config: ViTConfig, device: Device) -> Self {
        let c = config;
        assert!(c.image_size.is_multiple_of(c.patch_size), "image_size must be a multiple of patch_size");
        let param = |shape: &[usize]| (Tensor::randn(shape, device) * 0.02).detach().requires_grad_();
        ViT {
            patch_embed: Conv2d::new(c.in_channels, c.dim, c.patch_size, c.patch_size, 0, device),
            cls_token: param(&[1, 1, c.dim]),
            pos_embed: param(&[1, c.num_patches() + 1, c.dim]),
            blocks: (0..c.depth).map(|_| TransformerBlock::new(c.dim, c.heads, c.mlp_dim, c.dropout, false, device)).collect(),
            norm: LayerNorm::new(c.dim, device),
            head: Linear::new(c.dim, c.num_classes, device),
            config,
            training: AtomicBool::new(true),
        }
    }

    /// Patch + `[CLS]` token sequence after the encoder and final norm, `[N, 1 + patches, dim]`.
    pub fn encode(&self, x: &Tensor) -> Tensor {
        let s = x.shape();
        assert_eq!(
            (s[2], s[3]),
            (self.config.image_size, self.config.image_size),
            "ViT expects {0}x{0} images",
            self.config.image_size
        );
        let p = self.patch_embed.forward(x); // [N, D, H/p, W/p]
        let (n, d) = (p.shape()[0], p.shape()[1]);
        let tokens = p.reshape(&[n, d, self.config.num_patches()]).permute(&[0, 2, 1]); // [N, P, D]
        let cls = self.cls_token.expand(&[n, 1, d]);
        let mut h = Tensor::cat(&[&cls, &tokens], 1) + &self.pos_embed;
        h = F::dropout(&h, self.config.dropout, self.training.load(Ordering::Relaxed));
        for b in &self.blocks {
            h = b.forward(&h);
        }
        self.norm.forward(&h)
    }
}

impl Module for ViT {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.head.forward(&self.encode(x).select(1, 0))
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![("cls_token".to_string(), self.cls_token.clone()), ("pos_embed".to_string(), self.pos_embed.clone())];
        v.extend(prefixed("patch_embed", self.patch_embed.named_parameters()));
        for (i, b) in self.blocks.iter().enumerate() {
            v.extend(prefixed(&format!("blocks.{i}"), b.named_parameters()));
        }
        v.extend(prefixed("norm", self.norm.named_parameters()));
        v.extend(prefixed("head", self.head.named_parameters()));
        v
    }
    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
        self.blocks.iter().for_each(|b| b.set_training(training));
    }
}
