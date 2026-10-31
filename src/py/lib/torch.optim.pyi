# crate: pyrt/torch
# rust: ruai::pyrt::lib::torch::nn
"""py0 package `torch.optim` (`import torch.optim as optim`): see torch.pyi."""

from torch import Optimizer, Tensor

@rust("ruai::pyrt::lib::torch::nn::sgd")
def SGD(params: list[Tensor], lr: float, momentum: float = 0.0, weight_decay: float = 0.0) -> Optimizer: ...

@rust("ruai::pyrt::lib::torch::nn::adam")
def Adam(params: list[Tensor], lr: float = 0.001, weight_decay: float = 0.0) -> Optimizer: ...
