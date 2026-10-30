use crate::device::Device;
use crate::tensor::Tensor;

/// A neural-network building block (like `torch.nn.Module`).
///
/// Implementors provide `forward` and list their tensors via
/// `named_parameters` / `named_buffers`; container modules forward
/// `set_training` to their children.
pub trait Module: Send + Sync {
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Trainable tensors, with hierarchical names like `"0.weight"`.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    /// Non-trainable state saved with the model (e.g. BatchNorm running stats).
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    /// Switch between training and evaluation behaviour (dropout, batch norm).
    fn set_training(&self, _training: bool) {}

    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters().into_iter().map(|(_, t)| t).collect()
    }

    fn train(&self) {
        self.set_training(true);
    }

    fn eval(&self) {
        self.set_training(false);
    }

    /// Parameters and buffers, i.e. everything that is saved/loaded.
    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = self.named_parameters();
        v.extend(self.named_buffers());
        v
    }

    /// Move all parameters and buffers to `device` (in place).
    fn to_device(&self, device: Device) {
        for (_, t) in self.state_dict() {
            t.set_storage(t.storage().to_device(device));
            t.zero_grad();
        }
    }

    /// Convert all floating-point parameters and buffers to `dtype` in place
    /// (e.g. `F16` for half-precision inference). For mixed-precision
    /// *training* keep the model in f32 and use [`crate::autocast`] instead.
    fn to_dtype(&self, dtype: crate::DType) {
        assert!(dtype.is_float(), "Module::to_dtype needs a floating-point dtype");
        for (_, t) in self.state_dict() {
            if t.dtype().is_float() && t.dtype() != dtype {
                t.set_data(t.backend().cast(&t.storage(), dtype));
            }
        }
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

/// Prefix child names: `("weight", t)` -> `("fc1.weight", t)`.
pub fn prefixed(prefix: &str, items: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    items.into_iter().map(|(n, t)| (format!("{prefix}.{n}"), t)).collect()
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn forward(&self, x: &Tensor) -> Tensor {
        (**self).forward(x)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        (**self).named_parameters()
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        (**self).named_buffers()
    }
    fn set_training(&self, training: bool) {
        (**self).set_training(training)
    }
}

/// A chain of modules applied in order.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: `Sequential::new().add(a).add(b)`.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Box::new(m));
        self
    }

    pub fn push<M: Module + 'static>(&mut self, m: M) {
        self.layers.push(Box::new(m));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.iter().fold(x.clone(), |h, l| l.forward(&h))
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| prefixed(&i.to_string(), l.named_parameters())).collect()
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| prefixed(&i.to_string(), l.named_buffers())).collect()
    }
    fn set_training(&self, training: bool) {
        self.layers.iter().for_each(|l| l.set_training(training));
    }
}
