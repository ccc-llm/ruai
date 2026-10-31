# crate: pyrt/torch
# rust: ruai::pyrt::lib::torch::nn
"""py0 package `torch.nn` (`import torch.nn as nn`, `from torch import nn`): see torch.pyi."""

from torch import Loss, Module
from rust import Result

@rust("ruai::pyrt::lib::torch::nn::linear")
def Linear(in_features: int, out_features: int, bias: bool = True) -> Result[Module, RuntimeError]: ...

@rust("ruai::pyrt::lib::torch::nn::relu")
def ReLU() -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::sigmoid")
def Sigmoid() -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::tanh")
def Tanh() -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::softmax")
def Softmax(dim: int) -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::flatten")
def Flatten() -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::identity")
def Identity() -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::sequential")
def Sequential(*modules: Module) -> Module: ...

@rust("ruai::pyrt::lib::torch::nn::mse")
def MSELoss() -> Loss: ...

@rust("ruai::pyrt::lib::torch::nn::l1")
def L1Loss() -> Loss: ...

@rust("ruai::pyrt::lib::torch::nn::cross_entropy_loss")
def CrossEntropyLoss() -> Loss: ...
