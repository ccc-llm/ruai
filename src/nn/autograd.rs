//! Functional gradients (like `torch.autograd.grad`): gradients of an output
//! with respect to chosen tensors, returned instead of accumulated into
//! `.grad`, optionally differentiable themselves (`create_graph`) for second
//! derivatives, Hessian-vector products, gradient penalties or
//! meta-learning.
//!
//! ```
//! use ruai::{autograd, Device, Tensor};
//! let x = Tensor::from_vec(vec![1.0, 2.0], &[2], Device::Cpu).requires_grad_();
//! let y = (&x * &x * &x).sum_all(); // Σ x³
//! let dy = autograd::grad(&y, &[&x], true)[0].clone().unwrap(); // 3x²
//! let d2y = autograd::grad(&dy.sum_all(), &[&x], false)[0].clone().unwrap(); // 6x
//! assert_eq!(d2y.to_vec(), vec![6.0, 12.0]);
//! assert!(x.grad().is_none(), ".grad is untouched");
//! ```

use crate::nn::tensor::{run_backward, Tensor};

/// Gradients of `output` (seeded with ones) with respect to each of `inputs`
/// (`None` where an input does not influence `output`). With `create_graph`
/// the gradients carry an autograd graph and can be differentiated again.
/// No `.grad` is modified.
pub fn grad(output: &Tensor, inputs: &[&Tensor], create_graph: bool) -> Vec<Option<Tensor>> {
    grad_with(output, Tensor::ones_like(output), inputs, create_graph)
}

/// Like [`grad`] with an explicit seed gradient `d loss / d output` (a
/// vector-Jacobian product).
pub fn grad_with(output: &Tensor, seed: Tensor, inputs: &[&Tensor], create_graph: bool) -> Vec<Option<Tensor>> {
    if !output.requires_grad() {
        return vec![None; inputs.len()];
    }
    let inputs: Vec<Tensor> = inputs.iter().map(|&t| t.clone()).collect();
    run_backward(&[(output.clone(), seed)], create_graph, Some(&inputs))
}
