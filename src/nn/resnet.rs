//! Residual networks (He et al. 2016): ResNet-18/34 (basic blocks) and
//! ResNet-50/101/152 (bottleneck blocks), plus small CIFAR-style variants.
//!
//! Parameter names follow torchvision (`conv1.weight`, `layer1.0.bn2.bias`,
//! `layer2.0.downsample.0.weight`, `fc.weight`, ...), so the parameter count
//! of [`ResNetConfig::resnet18`] etc. matches torchvision exactly.
//! [`ResNet::load_torchvision`] converts a torchvision `state_dict` saved as
//! safetensors (batch-norm vectors are reshaped and `fc.weight` transposed).

use super::layers::{BatchNorm2d, Conv2d, Linear};
use super::module::{prefixed, Module};
use crate::device::Device;
use crate::functional as F;
use crate::serialize;
use crate::tensor::Tensor;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// Two 3x3 convolutions (ResNet-18/34).
    Basic,
    /// 1x1 -> 3x3 -> 1x1 with 4x channel expansion (ResNet-50+).
    Bottleneck,
}

impl BlockKind {
    pub fn expansion(&self) -> usize {
        match self {
            BlockKind::Basic => 1,
            BlockKind::Bottleneck => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResNetConfig {
    pub block: BlockKind,
    /// Number of blocks in each stage.
    pub layers: Vec<usize>,
    /// Base channel width of each stage (output is `width * expansion`).
    pub widths: Vec<usize>,
    pub in_channels: usize,
    pub num_classes: usize,
    /// ImageNet stem (7x7 stride-2 conv + 3x3 stride-2 max pool); otherwise
    /// a CIFAR stem (3x3 stride-1 conv, no pooling) for small images.
    pub imagenet_stem: bool,
}

impl ResNetConfig {
    fn imagenet(block: BlockKind, layers: &[usize], num_classes: usize) -> Self {
        ResNetConfig {
            block,
            layers: layers.to_vec(),
            widths: vec![64, 128, 256, 512],
            in_channels: 3,
            num_classes,
            imagenet_stem: true,
        }
    }
    pub fn resnet18(num_classes: usize) -> Self {
        Self::imagenet(BlockKind::Basic, &[2, 2, 2, 2], num_classes)
    }
    pub fn resnet34(num_classes: usize) -> Self {
        Self::imagenet(BlockKind::Basic, &[3, 4, 6, 3], num_classes)
    }
    pub fn resnet50(num_classes: usize) -> Self {
        Self::imagenet(BlockKind::Bottleneck, &[3, 4, 6, 3], num_classes)
    }
    pub fn resnet101(num_classes: usize) -> Self {
        Self::imagenet(BlockKind::Bottleneck, &[3, 4, 23, 3], num_classes)
    }
    pub fn resnet152(num_classes: usize) -> Self {
        Self::imagenet(BlockKind::Bottleneck, &[3, 8, 36, 3], num_classes)
    }
    /// A small basic-block ResNet for low-resolution images (CIFAR stem).
    pub fn small(in_channels: usize, widths: &[usize], blocks_per_stage: usize, num_classes: usize) -> Self {
        ResNetConfig {
            block: BlockKind::Basic,
            layers: vec![blocks_per_stage; widths.len()],
            widths: widths.to_vec(),
            in_channels,
            num_classes,
            imagenet_stem: false,
        }
    }
}

/// Convolution followed by batch norm.
struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm2d,
}

impl ConvBn {
    fn new(cin: usize, cout: usize, k: usize, stride: usize, device: Device) -> Self {
        ConvBn { conv: Conv2d::with_bias(cin, cout, k, stride, k / 2, false, device), bn: BatchNorm2d::new(cout, device) }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        self.bn.forward(&self.conv.forward(x))
    }
}

/// One residual block: `relu(convs(x) + shortcut(x))`.
pub struct ResBlock {
    /// conv1/bn1, conv2/bn2 (and conv3/bn3 for bottlenecks).
    convs: Vec<ConvBn>,
    downsample: Option<ConvBn>,
}

impl ResBlock {
    fn new(kind: BlockKind, cin: usize, width: usize, stride: usize, device: Device) -> Self {
        let cout = width * kind.expansion();
        let convs = match kind {
            BlockKind::Basic => {
                vec![ConvBn::new(cin, width, 3, stride, device), ConvBn::new(width, width, 3, 1, device)]
            }
            // torchvision ("ResNet v1.5") puts the stride on the 3x3 convolution
            BlockKind::Bottleneck => vec![
                ConvBn::new(cin, width, 1, 1, device),
                ConvBn::new(width, width, 3, stride, device),
                ConvBn::new(width, cout, 1, 1, device),
            ],
        };
        let downsample = (stride != 1 || cin != cout).then(|| ConvBn::new(cin, cout, 1, stride, device));
        ResBlock { convs, downsample }
    }
}

impl Module for ResBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let last = self.convs.len() - 1;
        let mut h = x.clone();
        for (i, c) in self.convs.iter().enumerate() {
            h = c.forward(&h);
            if i < last {
                h = h.relu();
            }
        }
        let shortcut = match &self.downsample {
            Some(d) => d.forward(x),
            None => x.clone(),
        };
        (h + shortcut).relu()
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.named(|m| m.named_parameters())
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.named(|m| m.named_buffers())
    }
    fn set_training(&self, training: bool) {
        self.convs.iter().chain(&self.downsample).for_each(|c| c.bn.set_training(training));
    }
}

impl ResBlock {
    fn named(&self, f: impl Fn(&dyn Module) -> Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        for (i, c) in self.convs.iter().enumerate() {
            v.extend(prefixed(&format!("conv{}", i + 1), f(&c.conv)));
            v.extend(prefixed(&format!("bn{}", i + 1), f(&c.bn)));
        }
        if let Some(d) = &self.downsample {
            v.extend(prefixed("downsample.0", f(&d.conv)));
            v.extend(prefixed("downsample.1", f(&d.bn)));
        }
        v
    }
}

/// A residual network for image classification: `[N, C, H, W]` -> logits `[N, classes]`.
pub struct ResNet {
    pub config: ResNetConfig,
    stem: ConvBn,
    stages: Vec<Vec<ResBlock>>,
    pub fc: Linear,
}

impl ResNet {
    pub fn new(config: ResNetConfig, device: Device) -> Self {
        assert_eq!(config.layers.len(), config.widths.len(), "layers and widths must have the same length");
        let w0 = config.widths[0];
        let stem = if config.imagenet_stem {
            ConvBn { conv: Conv2d::with_bias(config.in_channels, w0, 7, 2, 3, false, device), bn: BatchNorm2d::new(w0, device) }
        } else {
            ConvBn::new(config.in_channels, w0, 3, 1, device)
        };
        let mut cin = w0;
        let mut stages = Vec::new();
        for (i, (&n, &width)) in config.layers.iter().zip(&config.widths).enumerate() {
            let stride = if i == 0 { 1 } else { 2 };
            let blocks: Vec<ResBlock> = (0..n)
                .map(|j| {
                    let b = ResBlock::new(config.block, cin, width, if j == 0 { stride } else { 1 }, device);
                    cin = width * config.block.expansion();
                    b
                })
                .collect();
            stages.push(blocks);
        }
        ResNet { fc: Linear::new(cin, config.num_classes, device), config, stem, stages }
    }

    /// Feature map before global pooling, `[N, C_last, H', W']`.
    pub fn features(&self, x: &Tensor) -> Tensor {
        let mut h = self.stem.forward(x).relu();
        if self.config.imagenet_stem {
            h = F::max_pool2d_padded(&h, 3, 2, 1);
        }
        for stage in &self.stages {
            for b in stage {
                h = b.forward(&h);
            }
        }
        h
    }

    /// Load a torchvision `state_dict` stored as safetensors, e.g. exported with
    /// `safetensors.torch.save_file(torchvision.models.resnet18(weights=...).state_dict(), path)`.
    pub fn load_torchvision(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let (tensors, _) = serialize::load_safetensors(path)?;
        self.load_torchvision_state_dict(tensors)
    }

    /// See [`ResNet::load_torchvision`].
    pub fn load_torchvision_state_dict(&self, tensors: Vec<(String, Tensor)>) -> io::Result<()> {
        let tensors: Vec<(String, Tensor)> = tensors
            .into_iter()
            .filter(|(n, _)| !n.ends_with("num_batches_tracked"))
            .map(|(n, t)| {
                let t = if n == "fc.weight" {
                    t.t() // torch nn.Linear is [out, in]
                } else if t.rank() == 1 && n != "fc.bias" {
                    t.reshape(&[1, t.numel(), 1, 1]) // batch norm vectors
                } else {
                    t
                };
                (n, t)
            })
            .collect();
        serialize::load_state_dict(self, &tensors, true).map(|_| ())
    }
}

impl Module for ResNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc.forward(&F::adaptive_avg_pool_1x1(&self.features(x)))
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.named(|m| m.named_parameters())
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.named(|m| m.named_buffers())
    }
    fn set_training(&self, training: bool) {
        self.stem.bn.set_training(training);
        self.stages.iter().flatten().for_each(|b| b.set_training(training));
    }
}

impl ResNet {
    fn named(&self, f: impl Fn(&dyn Module) -> Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
        let mut v = prefixed("conv1", f(&self.stem.conv));
        v.extend(prefixed("bn1", f(&self.stem.bn)));
        for (i, stage) in self.stages.iter().enumerate() {
            for (j, b) in stage.iter().enumerate() {
                v.extend(prefixed(&format!("layer{}.{j}", i + 1), f(b)));
            }
        }
        v.extend(prefixed("fc", f(&self.fc)));
        v
    }
}
