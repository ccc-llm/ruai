# Changelog

## 3.2.0

- py2rs is merged into ruai with its full git history (`git subtree`) and becomes **py0**:
  the translator is `ruai::py` (feature `py`, on by default), the runtime `ruai::pyrt`
  (feature `pyrt`, plus `pyrt-re` for `re`). `ruai py run / emit / check / explain / build /
  pyo3 / stub / test / fuzz` replace the `py2rs` commands.
- Translated programs depend on `ruai` with `default-features = false, features = ["pyrt"]`
  (a path dependency in a checkout, the published version otherwise) and start with
  `use ruai::pyrt;`; the runtime's macros are `pyrt::print!`, `pyrt::println!`,
  `pyrt::check!` (hidden at the crate root).
- The py2rs corpus moved to `tests/py0/` and runs in `cargo test` (`tests/py0.rs`): all
  238 programs print what CPython 3.14 prints and the 44 rejections are reported as expected;
  the PyO3 module test passes. Docs moved to `py0/`.

## 3.1.0

- `ruai::world` (feature `world`, no dependencies): reinforcement-learning environments with
  the Gymnasium 1.x API (`make`, `reset(seed)`, `step`, `Space::seed / sample / contains`,
  `TimeLimit`, `RecordEpisodeStatistics`, `SyncVectorEnv` with next-step autoreset):
  CartPole-v1, MountainCar-v0, Pendulum-v1, Acrobot-v1, FrozenLake-v1, FrozenLake8x8-v1
  (with Gymnasium's `ansi` rendering). Seeded episodes match `gymnasium.make(id)` bit for bit
  (tests/world.rs compares 4 seeds x 400 steps of every environment, the vector env and
  the rendering against Gymnasium when it is installed).
- `ruai::rng`: NumPy's `SeedSequence`, `PCG64` and `Generator.random / uniform / integers`,
  bit-identical to NumPy (including its fused multiply-add on aarch64).
- `ruai world list`, `ruai world run ID [--episodes N] [--seed S] [--render]`.
- `examples/cartpole.rs`: REINFORCE with a `ruai::nn` policy solves CartPole-v1.

## 3.0.0

ruai becomes a Swiss-army toolkit: one crate whose parts are cargo features, and a git-style
`ruai` command.

- The engine moved to `src/nn/` (`ruai::nn`, feature `nn`, on by default) and is re-exported
  at the crate root, so v2 paths (`ruai::Tensor`, `ruai::nn::Linear`,
  `ruai::nn::llama::Llama`, `ruai::prelude`) keep working. The former `nn` module of layers
  is now `ruai::nn::modules`. The engine's dependencies are optional; the crate builds with
  `--no-default-features`.
- The `ruai` command: `ruai nn devices`, `ruai nn info PATH` (model summary from the
  safetensors header), `ruai nn generate --model DIR` (GPT-2 or Llama-family models with
  their tokenizer, int8 / int4, sampling options); `ruai help`, `ruai --version`; any other
  `ruai NAME` runs `ruai-NAME` from `PATH`, passing its exit code through.

## 2.0.1

Verified against real pretrained weights (GPT-2 small and SmolLM2-135M from Hugging Face).

- **Fix (MPS):** Metal's `tanh` returns NaN for arguments above ~45, so GELU and `tanh`
  produced NaN on large activations: real GPT-2 generated only `<|endoftext|>` on MPS. The
  argument is now clamped to ±15 (where tanh is already ±1 in f32); new extreme-input
  GPU-vs-CPU test.
- GPT-2: greedy generation matches Hugging Face's own integration test token for token on CPU
  and MPS. The int8 test now checks logit error (1.8% of the range) instead of an exact
  continuation (f32 prefers " was" over "'s" by only 0.33 logits there).
- SmolLM2-135M: logits match an independent plain-PyTorch `LlamaForCausalLM` to 3e-4; the
  real-weight test now uses those reference values instead of a guessed "Paris" continuation.

## 2.0.0

- Strided views: tensors are a buffer plus `(shape, strides, offset)`; `permute`, `transpose`,
  `t`, `narrow`, `select`, `expand` (stride 0), `unsqueeze`, `squeeze` and `reshape` of
  row-major data no longer copy. `Tensor::storage()` still returns contiguous data (a view is
  materialized once with the new `Backend::strided_copy` primitive and cached); new
  `is_contiguous`, `contiguous`, `strided_parts`.
- `matmul` reads transposed and batch-strided views in place; backends now honor
  `MatmulSpec` batch strides literally (MPS materializes strided batches: its GEMM ignores
  `matrixBytes`).
- Preallocated KV cache: `LayerKv` keeps `[B, H, capacity, D]` buffers that double when full;
  appends are written in place (new `Backend::write_slice`, in place when the buffer is not
  shared, otherwise copy on write) and attention reads a view of the filled part
  (`AttentionSpec::k_stride` / `v_stride`). Greedy decoding of 1024 tokens with a small Llama:
  about 1.4x faster on CPU and 6x on MPS. **Breaking:** `LayerKv::k` / `v` fields are replaced by
  `keys()` / `values()`; new `capacity()`; `clear()` keeps the buffers. When gradients flow
  through the cache it falls back to concatenation, keeping the graph.
- Llama grouped-query attention folds the query heads sharing a key/value head into extra query
  rows instead of repeating the keys and values.
- `Tensor::from_storage_strided`.
- Second-order gradients: `Tensor::backward_create_graph` and `autograd::grad` / `grad_with`
  (gradients of chosen tensors, returned instead of accumulated into `.grad`, optionally with
  their own graph); `ruai::enable_grad`. Every op's backward function switches to
  differentiable ops while a graph is being built (fused kernels otherwise), and
  `index_add` / `scatter_add` / `col2im` adjoints are differentiable ops. Gradient
  checkpointing and fused attention now return their gradients through `autograd::grad`
  instead of a nested `backward`.
- Test fixes: GPU reduction tolerance scales with the reduced length; the seq2seq training test
  allows for GPU atomic-add nondeterminism.

## 1.5.0

- `Backend::attention` primitive, used by `F::scaled_dot_product_attention` for `[B, H, T, D]`
  inputs without active dropout (any mask broadcastable to `[B, H, Tq, Tk]`); the backward pass
  recomputes with `F::attention_composed` (new public reference path).
  - CPU: FlashAttention-style tiling (64 x 128 score tiles, two small GEMMs per tile, online
    softmax); the `[Tq, Tk]` scores are never stored. 1.3-1.6x faster than the composed path
    on an M3 (up to 3.8x when the composed scores no longer fit in cache).
  - Other backends: `backend::attention_chunked`, query blocks of at most 16M scores with
    GEMM + softmax, so long sequences use bounded memory at GEMM speed.
- MPS int8 / int4 matvec kernels split `k` across 8 simdgroups: int8 is now about 4x and int4
  about 2x faster than the f32 GEMM for decoding (were slower).

## 1.4.0

- Group-wise int4 weight-only quantization: `Linear::quantize_int4(group)`,
  `Llama::quantize_int4`, `GPT2::quantize_int4` (output head stays int8). Two 4-bit values
  per byte plus one f32 scale per `group` input rows: about 1/8 of the f32 weight memory.
- Backend primitives `dequant_int4` / `matmul_int4` with fused kernels on CPU, MPS and CUDA
  (dequantize + GEMM for more than 16 rows); `Tensor::matmul_int4` is differentiable in its
  input, so LoRA trains on int4 models.
- `Linear::dims`; `--int4` in the `gpt2` and `llama` examples; quantized matvec benchmark.

## 1.3.0

- LoRA: `Linear::add_lora` / `merge_lora`, `Llama::add_lora` and `GPT2::add_lora` (by
  projection name, e.g. `&["q_proj", "v_proj"]`), `nn::lora_state_dict` to save only the
  adapters; works on int8-quantized models (QLoRA-style).
- Gradient checkpointing: `F::grad_checkpoint` (replays the RNG state and autocast mode) and
  `set_gradient_checkpointing` on `Llama`, `GPT2` and `GPT`; gradients are identical.
- `Module::trainable_parameters`, `freeze`, `unfreeze`; `ruai::enable_grad`.
- `matmul_int8` is now differentiable with respect to its input (needed to train adapters
  through int8 layers).
- Layers and transformer blocks are `Clone` (sharing parameters).
- The preallocated KV cache planned for this version moved to 2.0: without strided views
  every read of a preallocated buffer would copy it anyway.

## 1.2.0

- `tokenizer::HfTokenizer`: Hugging Face `tokenizer.json` for BPE models (GPT-2, Llama 2/3,
  TinyLlama, SmolLM, Qwen, ...): normalizers, pre-tokenizers (ByteLevel, Split regex with
  lookahead, Metaspace, Digits, ...), byte fallback, added / special tokens, post-processor
  templates (BOS), decoders. Unsupported components are reported when loading.
- `llama` example: `--model DIR --prompt TEXT` loads a pretrained checkpoint with its
  tokenizer and streams a greedy continuation; `scripts/download_smollm.sh`.
- Real-weight tests (skipped unless downloaded): `tokenizer.json` vs `vocab.json`/`merges.txt`
  for GPT-2, SmolLM2-135M continuing "The capital of France is" with "Paris".
- New dependencies: `fancy-regex`, `unicode-normalization`.

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
