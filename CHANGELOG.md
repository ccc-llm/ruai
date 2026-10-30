# Changelog

## 1.1.0

- GitHub Actions CI: Linux (CPU tests, clippy, rustdoc, CUDA build) and macOS (Accelerate, Metal);
  a manual run can also download GPT-2 and run the golden tests against the real weights.
- `Tensor::zeros_like` / `ones_like` keep the input's dtype (were always F32); new `Tensor::full_like`.
- `ClassificationData::collator` keeps the sample shape (`[B, C, H, W]` images);
  `collate` still flattens.
- `Rng::below(0)` panics with a clear message instead of dividing by zero.

## 1.0.0

API stabilization and documentation.

- **Breaking:** the crate is now `ruai` (package and library; was package `rust-nn`, library
  `rustnn`): replace `use rustnn::...` with `use ruai::...`. The `RUSTNN_DEVICE` environment
  variable keeps its name.
- **Breaking:** `GPT::generate(prompt, n, &mut Sampler)` now takes a `Sampler` (was a bare
  `temperature`), like `GPT2::generate` and `Llama::generate`. Use `Sampler::greedy()` for
  the old `temperature = 0.0`, `Sampler::new(t, 0, 1.0, seed)` for plain temperature sampling.
- `Sampler` is in the prelude.
- Every public item has rustdoc (`#![warn(missing_docs)]`); crate docs gained a training example.
- Cargo metadata for packaging (repository, keywords, categories, excludes).

## 0.9.0

- `nn::resnet`: ResNet-18/34/50/101/152 (basic / bottleneck blocks) and small CIFAR-style
  variants; torchvision parameter names and counts, `load_torchvision`.
- `nn::vit`: Vision Transformer; ViT-B/16 parameter count matches timm.
- `nn::seq2seq`: pre-LN encoder-decoder Transformer with cross-attention and cached greedy
  decoding; `F::padding_mask`, `F::cross_entropy_ignore`.
- `Conv2d::with_bias`, `MaxPool2d` padding (`F::max_pool2d_padded`), `GlobalAvgPool`.
- Examples: `cnn --model resnet|vit`, `seq2seq`.

## 0.8.0

- Llama building blocks: `F::rms_norm`, RoPE (`rope_inv_freq`, `rope_cos_sin`, `apply_rope`),
  `F::repeat_kv`, `Tensor::silu`, `nn::RMSNorm`, `nn::SwiGLU`.
- `nn::llama::Llama`: grouped-query attention, KV cache, tied/untied head, int8, Llama 3.1
  RoPE scaling; Hugging Face `from_pretrained` / `save_pretrained`.
- `generation::generate_cached`, shared by the language models.
- Fixed: `narrow` backward on half-precision tensors; `cat` promotes mixed dtypes.

## 0.7.0

- KV cache for GPT and GPT-2; `nn::gpt2::GPT2` loading Hugging Face weights unconverted.
- `tokenizer::Gpt2Tokenizer` (byte-level BPE), `generation::Sampler` (greedy / temperature /
  top-k / top-p).
- int8 weight-only quantization with fused `matmul_int8` kernels on CPU, MPS and CUDA.

## 0.6.0

- `F16` / `BF16` dtypes on every backend, `autocast`, `amp::GradScaler`, `Module::to_dtype`.

## 0.5.0

- safetensors I/O, `load_state_dict`, resumable checkpoints (model + optimizer + RNG),
  `DataLoader`, learning-rate schedulers, metrics, gradient accumulation.

## 0.4.0

- CUDA backend (cudarc: NVRTC kernels + cuBLAS), multi-GPU.

## 0.3.0

- Apple Metal (MPS) backend and cross-device tests.

## 0.2.0

- dtypes (I32 indices), new ops (cat / gather / where ...), fused primitives, fast CPU GEMM,
  Apple Accelerate.

## 0.1.0

- CPU backend, autograd, CNN, Transformer / GPT, optimizers, serialization.
