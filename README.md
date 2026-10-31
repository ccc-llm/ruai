# ruai (rust-nn)

[![CI](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml/badge.svg)](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml)

English | [繁體中文](README.tw.md)

A Swiss-army AI toolkit in Rust: one crate and one `ruai` command. Each part is a cargo feature, called like a git subcommand.

| Part | Command | What it is |
|---|---|---|
| `ruai::nn` (feature `nn`) | `ruai nn ...` | A PyTorch-like deep learning engine: dynamic graphs and autograd (with second derivatives), CPU / Apple GPU (MPS) / NVIDIA GPU (CUDA), CNN, ResNet, ViT, GPT / GPT-2 / Llama, quantization, safetensors |
| `ruai::world` (feature `world`) | `ruai world ...` | The Gymnasium 1.x API and environments (CartPole, MountainCar, Pendulum, Acrobot, FrozenLake), bit for bit identical to Gymnasium for the same seed |
| `ruai::py` (feature `py`) | `ruai py ...` | py0: a typed subset of Python, translated to readable Rust, compiled and run (merged from py2rs, see [py0/](py0/README.md)), with Python's standard library and subsets of numpy, scikit-learn, Gymnasium and PyTorch rewritten in Rust |
| agent (feature `agent`) | `ruai agent ...` | agge written in py0: teams of LLM agents that write code and books, graph workflows ([src/agent/](src/agent/README.md), which CPython runs too) |

```bash
cargo install ruai                    # or, in a checkout: cargo build --release
ruai nn devices                       # available compute devices
ruai nn info data/smollm2-135m        # model summary (reads only the safetensors header)
ruai nn generate --model data/gpt2 --prompt "The dog" -n 18
ruai nn generate --model data/smollm2-135m --device mps --int8 --prompt "Gravity is"
ruai py run prog.py                   # py0: translate to Rust, compile (cached) and run
ruai py emit prog.py                  # print the Rust translation
ruai world list                       # built-in environments and their spaces
ruai world run CartPole-v1 --episodes 3 --seed 1
ruai agent init && ruai agent chat    # LLM agent teams: interactive chat (/help for commands)
ruai agent code "write a todo CLI" -o todo   # design → implement → test → review
ruai NAME ...                         # any other NAME runs ruai-NAME from PATH (like git)
```

In the library the engine is `ruai::nn`, also re-exported at the crate root (`ruai::Tensor` is `ruai::nn::Tensor`; v2 paths such as `ruai::nn::Linear` and `ruai::nn::llama::Llama` keep working). To use only some parts, turn off the default features: `ruai = { version = "4", default-features = false, features = [...] }`.

## Installation

```toml
[dependencies]
ruai = "4"     # or from git: ruai = { git = "https://github.com/ccc-llm/ruai" }
# GPU: features = ["mps"] (Apple) or ["cuda"] (NVIDIA); macOS CPU acceleration: ["accelerate"]
```

> The crate is named [`ruai`](https://crates.io/crates/ruai) (`rust-nn` and `rustnn` were taken on crates.io); code imports it as `use ruai::...`. Changes per version are in [CHANGELOG.md](CHANGELOG.md).

## Quick start

```bash
cargo test                              # numerical gradient checks + end-to-end training tests
cargo run --release --example cnn       # CNN image classification (MNIST if present), a checkpoint per epoch
cargo run --release --example cnn -- --resume          # resume from runs/cnn/last.safetensors
cargo run --release --example cnn -- --model resnet    # small ResNet (cnn / resnet / vit)
cargo run --release --example cnn -- --model vit       # Vision Transformer (4×4 patches)
cargo run --release --example seq2seq   # an encoder-decoder Transformer learns to sort digit strings (padding mask + cached greedy)
cargo run --release --example gpt       # character-level GPT: train → save → load → generate
cargo run --release --example gpt -- --accum 4         # gradient accumulation (4 micro-batches per step)
cargo run --release --example bench     # benchmarks

# macOS: Apple Accelerate (AMX) for matrix products, about 2x faster on large models
cargo run --release --features accelerate --example gpt

# Apple GPU (Metal): add --features mps and pick the device with --device (cpu / mps / auto)
cargo run --release --features mps --example gpt -- --device mps
cargo run --release --features mps --example bench mps

# NVIDIA GPU (CUDA): add --features cuda
cargo run --release --features cuda --example gpt -- --device cuda

# pretrained GPT-2 (Hugging Face weights): download once, then generate text
./scripts/download_gpt2.sh
cargo run --release --example gpt2 -- --prompt "The meaning of life is"
cargo run --release --example gpt2 -- --int8                  # int8 weights
cargo run --release --example llama     # character-level Llama (RMSNorm/RoPE/SwiGLU/GQA) → export HF format → reload → generate
cargo run --release --features mps --example gpt2 -- --device mps

# mixed precision training (autocast + f32 master weights; f16 scales the loss automatically)
cargo run --release --features mps --example gpt -- --device mps --amp f16
```

Picking a device in code: `Tensor::randn(&[2, 3], Device::Mps)`, `model.to_device(Device::Mps)`, `Device::best_available()`, or the environment variable `RUSTNN_DEVICE=mps`.

MNIST: put the unpacked `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` in `data/mnist/`; without them the `cnn` example uses a generated data set of geometric shapes.

## Examples

```rust
use ruai::prelude::*;
use ruai::nn::{Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear};
use ruai::optim::Adam;

let dev = Device::Cpu;
let model = Sequential::new()
    .add(Conv2d::new(1, 8, 3, 1, 1, dev))
    .add(BatchNorm2d::new(8, dev))
    .add(ReLU)
    .add(MaxPool2d::new(2, 2))
    .add(Flatten)
    .add(Linear::new(8 * 14 * 14, 10, dev));

let mut opt = Adam::new(model.parameters(), 1e-3);
// each batch:
opt.zero_grad();
let loss = F::cross_entropy(&model.forward(&x), &labels, 0.0);
loss.backward();
opt.step();

// inference
model.eval();
let pred = no_grad(|| model.forward(&x).argmax(-1, false));

// save / load (safetensors, interoperable with PyTorch / Hugging Face)
ruai::serialize::save("model.safetensors", &model)?;
ruai::serialize::load("model.safetensors", &model)?;
```

### Data loading, schedules and checkpoints

```rust
use ruai::data::{ClassificationData, DataLoader};
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::checkpoint::{self, TrainState};

let collate = train_data.collator(dev);                 // [B, C, H, W] (collate flattens to [B, C*H*W])
let loader = DataLoader::new(train_data, 64, collate)
    .shuffle(true).drop_last(true).num_workers(2);    // background threads prepare batches
let sched = CosineLr { base_lr: 1e-3, min_lr: 1e-4, warmup: 100, total: 10_000 };
let mut state = TrainState::default();               // or checkpoint::load(...) to resume

for epoch in state.epoch..epochs {
    for batch in loader.epoch(epoch) {                // each epoch's shuffle is deterministic
        sched.apply(&mut opt, state.step as usize);
        opt.zero_grad();
        F::cross_entropy(&model.forward(&batch.inputs), &batch.targets, 0.0).backward();
        opt.step();
        state.step += 1;
    }
    state.epoch += 1;
    checkpoint::save("runs/last.safetensors", &model, &opt, &state)?;  // model + optimizer + RNG
}
```

### Pretrained GPT-2

```rust
use ruai::nn::gpt2::GPT2;
use ruai::tokenizer::Gpt2Tokenizer;
use ruai::generation::Sampler;

let tok = Gpt2Tokenizer::from_dir("data/gpt2")?;
let mut model = GPT2::from_pretrained("data/gpt2", Device::Cpu)?;  // reads HF's model.safetensors directly
model.quantize_int8();                                              // optional: int8 weights
let mut sampler = Sampler::new(0.8, 40, 0.95, 42);                  // temperature, top-k, top-p, seed
let ids = model.generate(&tok.encode("Once upon a time"), 50, &mut sampler);  // with a KV cache
println!("{}", tok.decode(&ids));
```

> ✅ Verified with the official weights (`./scripts/download_gpt2.sh`, then `cargo test --release --test inference`): the tokenizer's golden values, and greedy generation of 20 tokens after "The dog" identical to the GPT-2 integration test of Hugging Face transformers (`test_lm_generate_gpt2`), on CPU and MPS; int8 logits are off by about 1.8% of the logit range. Measured (Apple M3, 100 tokens): CPU f32 / int8 / int4 = 41 / 62 / 88 tokens/s, MPS = 58 / 163 / 178 tokens/s.

### Llama (RMSNorm / RoPE / SwiGLU / GQA)

```rust
use ruai::nn::llama::{Llama, LlamaConfig};

let cfg = LlamaConfig::new(vocab, 256, 4, 8, 2, 512);   // vocab, hidden, layers, heads, kv_heads, max_len
let model = Llama::new(cfg, Device::Cpu);               // 8 query heads share 2 K/V heads (GQA)
let logits = model.forward_tokens(&tokens);             // [B, T] -> [B, T, vocab]
model.save_pretrained("runs/llama")?;                    // config.json + model.safetensors (HF layout)
let model = Llama::from_pretrained("data/tinyllama", Device::Cpu)?;  // HF LlamaForCausalLM weights
let ids = model.generate(&prompt_ids, 50, &mut Sampler::greedy());   // KV cache (kv_heads copies only)
```

The parts can be used on their own: `F::rms_norm`, `F::rope_inv_freq` / `F::rope_cos_sin` / `F::apply_rope` (HF's rotate_half layout), `F::repeat_kv`, `Tensor::silu`, `nn::RMSNorm`, `nn::SwiGLU`. `config.json` supports `num_key_value_heads`, `head_dim`, `rope_theta`, `tie_word_embeddings`, `attention_bias` and Llama 3.1's `rope_scaling` (`rope_type: "llama3"`; other types are reported as errors). HF's `nn.Linear` weights are `[out, in]` and ruai's `Linear` is `[in, out]`: `from_pretrained` / `save_pretrained` transpose them.

Pretrained models (`config.json` + `model.safetensors` + `tokenizer.json`):

```bash
./scripts/download_smollm.sh                    # HuggingFaceTB/SmolLM2-135M → data/smollm2-135m
cargo run --release --example llama -- --model data/smollm2-135m --prompt "The capital of France is"
```

```rust
use ruai::tokenizer::HfTokenizer;
let tok = HfTokenizer::from_dir("data/smollm2-135m")?;          // reads tokenizer.json
let ids = tok.encode("Hello world", true);                      // true: add BOS and other special tokens
println!("{}", tok.decode(&ids, true));                         // true: skip special tokens
```

`HfTokenizer` supports BPE models (GPT-2, Llama 2/3, TinyLlama, SmolLM, Qwen, ...): normalizers, pre-tokenizers (ByteLevel, Split regexes with lookahead, Metaspace, Digits, ...), byte fallback, added / special tokens, post-processor templates and decoders; unsupported parts (WordPiece, Unigram, ...) are reported when loading.

> ✅ Verified with the official SmolLM2-135M weights (`./scripts/download_smollm.sh`, then `cargo test --release --test llama --test tokenizer`): `tokenizer.json` tokenizes correctly, and the logits differ by less than 3e-4 from an independent pure-PyTorch implementation of Hugging Face's `LlamaForCausalLM` (the same argmax at every position, CPU and MPS); GPT-2's `tokenizer.json` and `vocab.json` + `merges.txt` agree token for token. Measured (Apple M3, 40 tokens): CPU f32 / int8 / int4 = 36 / 53 / 61 tokens/s, MPS = 29 / 92 / 99 tokens/s (this small 30-layer model is bound by per-op dispatch on MPS). Not yet verified: Llama 3's `rope_scaling`, Llama 2's SentencePiece-style `tokenizer.json`.

### py0: a typed subset of Python (py)

```python
from dataclasses import dataclass

@dataclass
class Item:
    name: str
    price: float

def total(items: list[Item], tax: float = 0.05) -> float:
    return sum(i.price for i in items) * (1 + tax)

print(f"{total([Item('apple', 1.5), Item('pear', 2.25)]):.2f}")
```

`ruai py run prog.py` translates it to readable Rust (`ruai py emit` shows it), compiles it with Cargo (a few seconds the first time, cached after) and runs it; it prints what CPython prints, and what it cannot do it rejects when translating, with the line and a hint. py0 was merged from [py2rs](py0/README.md) with its full git history; the language is specified in [py0/SPEC.md](py0/SPEC.md). Translated programs depend on `ruai` with only runtime features (`default-features = false, features = ["pyrt", ...]`), so they do not compile the neural network engine. Running py0 programs needs a Rust toolchain (cargo).

**Packages and the standard library**: `import M` / `import M as A` / `from M import x` look for a package stub `src/py/lib/M.pyi` (and in the directories of `PY0_LIB`): typed declarations of functions, classes and constants, implemented in Rust under `ruai::pyrt::lib::M` (or the path a `# rust:` line gives; `# crate: pyrt/FEATURE` turns on a runtime feature of ruai). A parameter typed `A | B` makes overloads (a generic Rust function), several `def`s of one name are overloads too, and `Result[T, OSError | ...]` raises CPython's exception classes. Python's standard library is rewritten module by module in `src/pyrt/lib/`, each module checked against CPython (`tests/py0/py/std_*.py`): `shlex`, `fnmatch`, `html`, `subprocess.run`, more of `os` / `os.path` / `sys` / `time`, besides what py2rs built in (`math`, `collections`, `heapq`, `bisect`, `itertools`, `functools`, `random`, `re`, `sys`, `time`, `json.dumps`, `os.path`). ruai's own packages: `ruai.json` (`Json`, a JSON value with a static type whose `dumps` matches CPython's `json.dumps`) and `ruai.http` (HTTP(S) requests, read line by line for server-sent events); they have CPython stand-ins in `py0/python/ruai/`, so the same program runs on both.

### py0 with numpy

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.random((50, 3))
y = x @ np.array([2.0, -1.0, 0.5]) + 0.3
w = np.zeros(3)
for _ in range(2000):
    err = x @ w - y
    w = w - 0.5 * (x.T @ err) / x.shape[0]
print(np.round(w, 3), x[x[:, 0] > 0.9].shape, round(float(np.abs(x @ w - y).mean()), 6))
```

In py0, `import numpy as np` is a subset of NumPy written in Rust (`src/pyrt/lib/numpy/`): float64 / float32 / int64 / bool arrays, NumPy 2's type promotion and broadcasting, slices that are views sharing memory, boolean masks and index arrays, reductions over axes (`sum` adds in NumPy's pairwise order), `linalg`, and `random.default_rng` bit for bit as NumPy; `print` / `repr` match NumPy's `arrayprint` character for character. Under CPython the same program runs the real NumPy, and `tests/py0/numpy/` compares them. Because py0 is statically typed: `a[i]` and reductions give 0-d arrays (NumPy scalars), so write `float(a.sum())` where a Python float is needed; `a.shape` can be indexed but not unpacked; dtypes are strings (`dtype="int64"`). Floating-point matrix products add in order and may differ from NumPy's BLAS in the last bit.

### py0 with scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
sc = StandardScaler().fit(Xtr)
model = LogisticRegression().fit(sc.transform(Xtr), ytr)
print(model, model.score(sc.transform(Xte), yte))
```

In py0, `from sklearn.X import Y` is a subset of scikit-learn in Rust, on the numpy arrays (`src/pyrt/lib/sklearn/`): `LinearRegression`, `Ridge`, `LogisticRegression`, `KNeighborsClassifier` / `Regressor`, `GaussianNB`, `KMeans`, `PCA`, `StandardScaler`, `MinMaxScaler`, `train_test_split`, the common metrics and `load_iris`. The algorithms follow scikit-learn (the same centering, the solvers' optimum, PCA's sign convention, and `train_test_split` with the same `RandomState` numbers, so splits are identical); `tests/py0/sklearn/` compares the output with scikit-learn. Differences: iterative solvers that scikit-learn stops early (lbfgs) agree to their tolerance; k-NN may pick another neighbor when distances tie exactly; `KMeans` needs its initial centers (`init=array`); `target_names` is a list.

### py0 with Gymnasium and PyTorch

```python
import gymnasium as gym
import torch
import torch.nn as nn

torch.manual_seed(0)
policy = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
logits = policy(torch.from_numpy(obs))
obs, reward, terminated, truncated, info = env.step(int(logits.argmax()))
print(policy, logits, obs, reward)
```

`import gymnasium as gym` is `ruai::world`'s environments with Gymnasium's Python API (observations are float32 numpy arrays, bit for bit as Gymnasium's). `import torch` is a subset of PyTorch on ruai's nn engine: autograd, `nn.Linear` and other modules, `nn.Sequential`, losses, `optim.SGD` / `Adam`, `torch.no_grad()`. `torch.manual_seed` reproduces PyTorch's random number generator (MT19937), so `rand`, `randn` and `nn.Linear`'s initial weights are PyTorch's; tensors and modules print as in PyTorch. `tests/py0/gymnasium/` and `tests/py0/torch/` compare with Gymnasium and PyTorch (training loops agree at PyTorch's default 4 decimals). Differences: models are `nn.Sequential` or plain classes holding layers (subclassing `nn.Module` is not supported) and `parameters()` returns a list; dtypes are strings in py0; matrix products and element functions may differ from PyTorch's BLAS / vectorized code in the last bit.

### Plots: ruai.plot

```python
import numpy as np
import ruai.plot as plt

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), "--", label="cos")
ax.legend()
plt.show()          # opens the figure in the browser, drawn on a <canvas>
```

`ruai.plot` (`ruai::plot` in Rust, feature `plot`, no dependencies) has matplotlib.pyplot's API: `plot` (format strings such as `"r--o"`), `scatter`, `bar` (also categorical), `hist`, `imshow`, `text`, `axhline` / `axvline`, titles, labels, limits, legends, grids, `subplots` / `subplot`, and `Figure` / `Axes` / `Line2D` objects. Autoscaling and ticks are matplotlib's (5% margins, sticky edges, `MaxNLocator`), so `get_xlim()` / `get_xticks()` give matplotlib's numbers; `tests/py0/plot/` compares them. `plt.show()` writes an HTML page with a canvas and opens it in the browser (nothing with `MPLBACKEND=agg`); `savefig` writes `.svg` or `.html`. Under CPython the same program runs on matplotlib itself (`py0/python/ruai/plot.py`). Not yet: PNG output, 2-D grids from `subplots(n, m)` (use `subplot`), colorbars, log scales.

### Any Python library: the Python bridge

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6.28, 50)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(x, np.sin(x), "r-", label="sin")
ax.legend()
fig.savefig("sin.png")
```

Besides the packages rewritten in Rust, py0 can call **the real Python library**: a stub whose first line is `# python: MODULE` declares the module's functions and classes with py0 types, and each call goes to CPython embedded in the program (PyO3, runtime feature `pyrt-python`). Arguments and results convert both ways (int, float, bool, str, None, list, dict, tuple, numpy arrays); `= ...` as a default leaves the argument to Python's default; Python's exceptions become py0 exceptions with the same names. `matplotlib` and `matplotlib.pyplot` ship as curated stubs. For any other module, `ruai py stub --python MODULE` drafts one from `inspect` into `py0lib/MODULE.pyi` (looked up next to the program, in `./py0lib` and in `PY0_LIB`): annotated types map to py0's, unknown ones become `Object` (any Python object, accepting any value), so tighten the types by hand. The program is linked to the first Python that imports its modules (`PY0_PYTHON`, else python3.14 / 3.13 / 3.12 / python3) and uses that interpreter's packages at runtime. The tradeoffs: the program needs that Python installed to run, calls cost CPython's speed, and type checking reaches only as far as the stub. `tests/py0/pybridge/` compares bridged programs (matplotlib, and standard modules as examples) with CPython.

### LLM agent teams (agent)

```bash
ruai agent init                                   # create .agge/ here (settings, agents, graphs, logs)
ruai agent model add glm nvidia z-ai/glm-5.3-flash   # providers: openai, nvidia, openrouter, deepseek, groq, gemini, ollama, lmstudio, anthropic
ruai agent chat --team code                       # talk to the lead, who delegates (<delegate>) to architect / coder / tester / reviewer
ruai agent ask -y "create hello.py and run it"    # one task: the agent calls tools by text tags (read/write files, search, shell)
ruai agent book "Git for beginners" --chapters 3 -o gitBook   # outline → chapters (a map node) → editor
ruai agent run brainstorm "how to design a cache layer" -M merge=claude
```

`ruai agent` is [agge](src/agent/README.md) (about 2,500 lines of Python) rewritten in py0, translated to Rust and built into the `ruai` binary, so it needs neither Python nor cargo. The same sources (`src/agent/*.py`) run directly under CPython; `tests/agent.rs` runs the same commands on both (also against a fake OpenAI / Anthropic streaming server) and requires identical output, and checks that the embedded translation `src/cli/agent_gen.rs` is up to date. Differences from the original: parallel delegations, graph nodes and map items run one after another; the chat has no readline history or completion.

### Reinforcement-learning environments (world)

```rust
use ruai::world::{self, Value};

let mut env = world::make("CartPole-v1")?;            // like gymnasium.make: with TimeLimit(500)
env.action_space_mut().seed(1);
let (mut obs, _) = env.reset(Some(42));               // like env.reset(seed=42)
loop {
    let a = env.action_space_mut().sample();          // Value::Int / Value::Floats
    let s = env.step(&a);                             // obs, reward, terminated, truncated, info
    if s.terminated || s.truncated { break }
    obs = s.obs;
}
```

The dynamics, the order of floating-point operations and the random numbers (`ruai::rng`: NumPy's `SeedSequence` + `PCG64`) are Gymnasium 1.x's, so the same seed gives bit for bit the same observations, rewards and flags (`tests/world.rs` compares step by step when Gymnasium is installed). There are also `RecordEpisodeStatistics` and `SyncVectorEnv` (Gymnasium 1.x's next-step autoreset). `cargo run --release --example cartpole` solves CartPole with REINFORCE and a `ruai::nn` policy (about 40 updates).

### Fine-tuning: LoRA and gradient checkpointing

```rust
let mut model = Llama::from_pretrained("data/smollm2-135m", dev)?;
model.add_lora(8, 16.0, &["q_proj", "v_proj"]);       // freeze the model, train only low-rank adapters (rank 8, alpha 16)
model.set_gradient_checkpointing(true);              // recompute each layer's activations in backward to save memory
let mut opt = Adam::adamw(model.trainable_parameters(), 1e-4, 0.0);
// ...train...
serialize::save_tensors("adapter.safetensors", &nn::lora_state_dict(&model))?;  // save only the (small) adapters
model.merge_lora();                                   // merge them into the weights, then save_pretrained exports HF format
```

`add_lora` also works after `quantize_int8()` (QLoRA style: int8 base + f32 adapters). `F::grad_checkpoint(inputs, params, f)` works on any sub-network; recomputation restores the RNG state and the autocast mode, so gradients with dropout are exactly those without checkpointing.

### ResNet / ViT / encoder-decoder

```rust
use ruai::nn::resnet::{ResNet, ResNetConfig};
use ruai::nn::vit::{ViT, ViTConfig};
use ruai::nn::seq2seq::{Seq2Seq, Seq2SeqConfig};

let resnet = ResNet::new(ResNetConfig::resnet50(1000), dev);         // the same parameter count as torchvision
resnet.load_torchvision("resnet50.safetensors")?;                    // a torchvision state_dict (saved as safetensors)
let small = ResNet::new(ResNetConfig::small(3, &[16, 32, 64], 2, 10), dev);  // CIFAR-style stem
let vit = ViT::new(ViTConfig::base_16(1000), dev);                   // ViT-B/16: 86,567,656 parameters (as timm)

let s2s = Seq2Seq::new(cfg, dev);
let mask = F::padding_mask(&src, PAD);                               // [B, 1, 1, Ts]
let logits = s2s.forward_pair(&src, &tgt_in, Some(&mask));           // teacher forcing
let loss = F::cross_entropy_ignore(&logits.reshape(&[b * t, v]), &tgt_out, PAD);
let out = s2s.greedy(&src_ids, BOS, Some(EOS), 50);                  // decoder self-attention with a KV cache
```

### Second derivatives

```rust
use ruai::autograd;

// gradient penalty (WGAN-GP): differentiate the gradient with respect to the input again, by the parameters
let x = x.detach().requires_grad_();
let gx = autograd::grad(&d.forward(&x).sum_all(), &[&x], true)[0].clone().unwrap();  // create_graph = true
let norm = (&gx * &gx).sum(1, false).sqrt();
let penalty = (norm.add_scalar(-1.0).powf(2.0)).mean_all();
penalty.backward();                                   // gradients reach d's parameters

// or like PyTorch's backward(create_graph=True)
loss.backward_create_graph();                         // x.grad() itself is differentiable afterwards
```

With `create_graph` each operation's backward uses differentiable operations (the fused kernels otherwise); `tests/second_order.rs` checks every operation's Hessian-vector products against finite differences.

### Mixed precision (AMP)

```rust
use ruai::amp::{autocast, GradScaler};

let mut scaler = GradScaler::new();                   // needed for f16; optional for bf16
opt.zero_grad();
let loss = autocast(DType::F16, || F::cross_entropy(&model.forward(&x), &y, 0.0));
scaler.scale(&loss).backward();                       // parameters stay f32 and get f32 gradients
scaler.unscale(&opt);                                 // (when clipping with clip_grad_norm)
scaler.step(&mut opt);                                // skips the step and lowers the scale on overflow

// pure half-precision inference
model.to_dtype(DType::F16);
let y = model.forward(&x.to_dtype(DType::F16));
```

Autocast rules (as PyTorch): matmul / Linear / Conv run in half precision; softmax, log_softmax, LayerNorm, BatchNorm and losses always in f32; adding different float types promotes to f32, but a rank-0 scalar takes the other operand's type (`half * 2.0` is still half).

Gradient accumulation: `.grad` accumulates across `backward()` calls; multiply each micro-batch's loss by `1/k`, call `backward()` on each, then `step()` once (`tests/io.rs` checks that this equals the full batch).

## Features

| Area | Contents |
|---|---|
| DType | `F32`, `F16`, `BF16` (all differentiable; half precision computes in f32 and rounds), `I32` (indices: token ids, class labels, argmax results, kept on the device); differentiable `to_dtype`, type promotion rules |
| Inference | **KV cache** (a preallocated buffer that doubles its capacity; new tokens are written in place and attention reads views, nothing is copied per step; prefill + step-by-step decoding, a sliding window beyond the context length, copy on write after clone), `Sampler` (greedy / temperature / top-k / top-p, seedable), `matmul_nt` (a product with a transposed operand without copying, for tied heads) |
| GPT-2 | `nn::gpt2::GPT2`: parameter names as Hugging Face's `GPT2LMHeadModel`, and HF's Conv1D layout `[in, out]` is ruai's `Linear`, so `model.safetensors` loads unconverted; small/medium/large/xl; `tokenizer::Gpt2Tokenizer` (byte-level BPE, GPT-2's pre-tokenizer regex written by hand) |
| Llama | `nn::llama::Llama`: RMSNorm, RoPE (with Llama 3.1 frequency scaling), SwiGLU, GQA (the KV cache keeps only the K/V heads), tied / untied head, int8; parameter names as HF's `LlamaForCausalLM`, `from_pretrained` (one file or sharded safetensors) / `save_pretrained` |
| Vision | `nn::resnet::ResNet` (BasicBlock / Bottleneck, resnet18/34/50/101/152, a small CIFAR-style variant; parameter names and counts as torchvision, `load_torchvision` converts the BN shapes and transposes fc), `nn::vit::ViT` (patch embedding, `[CLS]`, learned positions; ViT-B/16's parameter count as timm) |
| seq2seq | `nn::seq2seq::Seq2Seq` (pre-LN encoder + a decoder with cross-attention, `encode` / `decode` / `decode_cached` / `greedy`), `DecoderBlock`, `F::padding_mask`, `F::cross_entropy_ignore` |
| int4 | `Linear::quantize_int4(group)`, `Llama` / `GPT2::quantize_int4`: symmetric int4 with one scale per `group` input rows, two values per byte (about 1/8 of f32); `matmul_int4` has fused kernels reading packed weights on CPU / MPS / CUDA; the output head stays int8 |
| int8 | `DType::I8`, `Linear::quantize_int8` (symmetric, per output channel), `GPT2::quantize_int8`; the `matmul_int8` primitive has fused kernels reading int8 on CPU / MPS / CUDA (1/4 of f32's memory traffic), and dequantizes + GEMM for large batches |
| Mixed precision | `autocast(F16 \| BF16, ..)`, `GradScaler` (dynamic loss scaling, skipped steps on overflow, state saved in checkpoints), `Module::to_dtype` |
| Tensor | any number of dimensions, NumPy broadcasting, `reshape/view/permute/transpose/unsqueeze/flatten/expand`, operator overloading (`&a + &b`, `a * 2.0`) |
| Autograd | dynamic graphs, backward in topological order, gradient accumulation, `no_grad()` / `NoGradGuard` / `enable_grad()` / `set_grad_enabled`, `detach()`; **second derivatives**: `backward_create_graph()`, `autograd::grad(output, inputs, create_graph)` (leaves `.grad` alone: Hessian-vector products, gradient penalties, MAML) |
| Operations | `+ - * /`, `exp log sqrt powf abs tanh sigmoid relu gelu silu sin cos maximum`, `sum mean var max min argmax`, `minimum clamp where_cond masked_fill`, batched `matmul`, `index_select gather narrow select cat stack`, `im2col` |
| functional | `softmax`, `log_softmax`, `cross_entropy` (with label smoothing), `mse_loss`, `bce_with_logits`, `conv2d`, `max/avg_pool2d`, `max_pool2d_padded`, `layer_norm`, `rms_norm`, `dropout`, `scaled_dot_product_attention`, `causal_mask`, sinusoidal positions, RoPE (`rope_cos_sin` / `apply_rope`), `repeat_kv` |
| nn | `Linear`, `RMSNorm`, `SwiGLU`, `Conv2d` (optionally without bias), `MaxPool2d` (with padding), `GlobalAvgPool`, `AvgPool2d`, `BatchNorm2d`, `LayerNorm`, `Embedding`, `Dropout`, `Flatten`, `ReLU/GELU/Tanh/Sigmoid`, `Sequential`, `FeedForward`, `MultiHeadAttention` (self / cross attention), `TransformerBlock` (pre-LN, optionally causal), `GPT` |
| optim | `SGD` (momentum, weight decay), `Adam`, `AdamW` (fused kernel, one pass per parameter), `clip_grad_norm`, `cosine_lr` (with warmup) |
| Data | `Dataset` trait, `DataLoader` (multi-threaded prefetch, a deterministic shuffle per epoch, `drop_last`, custom collate, early exit without hanging, worker panics propagate), `VecDataset`, `ClassificationData`, `Subset`, `random_split` |
| Saving | **safetensors** (Hugging Face's implementation): reads and writes F32/F16/BF16/I32 (keeping the type), also reads F64/I64/I16/I8/U8/U16/U32/BOOL; `load_state_dict` (strict, or reporting missing / unexpected); still reads v0.1–v0.4's RSNN format |
| Checkpoints | model + optimizer state (Adam's m/v/step, SGD's velocity) + step/epoch + the global RNG state + custom metadata, written to a temporary file and renamed; **resumed training is bit for bit identical to uninterrupted training** |
| Schedules / metrics | `ConstantLr`, `StepLr`, `ExponentialLr`, `CosineLr`, `LinearLr` (all with warmup; pure functions, so training resumes at any step); `RunningMean`, `Ema`, `Accuracy`, `Throughput` |
| Other | `train()/eval()` modes, seedable random numbers (`rand::Rng` independent streams) |

## Architecture

```
src/
├── main.rs, cli/    the `ruai` command (dispatch; ruai nn / world / py / agent)
├── nn/              the neural network engine (feature nn)
├── world/           Gymnasium environments (feature world)
├── rng.rs           NumPy-compatible random numbers, MT19937
├── py/              the py0 translator (feature py, formerly py2rs)
│   └── lib/         package stubs (*.pyi): standard library, numpy, sklearn, gymnasium, torch
├── pyrt/            the runtime of py0 programs (feature pyrt)
│   └── lib/         the packages in Rust (pyrt-numpy, pyrt-sklearn, pyrt-gymnasium, pyrt-torch, pyrt-plot, ...); python.rs: the Python bridge (pyrt-python)
└── agent/           `ruai agent` in py0 (translated into cli/agent_gen.rs)
nn/
├── backend/
│   ├── mod.rs     the Backend trait: about 15 "primitives", all a device has to implement
│   ├── cpu.rs     the CPU implementation (parallel with rayon)
│   ├── mps.rs     the Apple Metal backend (--features mps)
│   ├── mps_kernels.metal  Metal compute kernels (compiled at run time)
│   ├── cuda.rs    the NVIDIA CUDA backend (--features cuda)
│   └── cuda_kernels.cu    CUDA kernels (compiled at run time by NVRTC)
├── tensor.rs      Tensor, graph nodes, backward()
├── ops.rs         differentiable operations: forward calls the backend, backward is made of other operations
├── functional.rs  composite operations (softmax, conv2d, attention, ...)
├── modules/       the Module trait and layers, Transformer, GPT, GPT-2, Llama, ResNet, ViT, Seq2Seq
├── optim/         optimizers (mod.rs) and learning-rate schedules (lr.rs)
├── data.rs        Dataset / DataLoader
├── checkpoint.rs  resumable checkpoints
├── metrics.rs     training metrics
└── serialize.rs   safetensors reading and writing, load_state_dict
```

**Core design: everything rests on a few backend primitives.** The `Backend` trait has two layers:

- **Required primitives** (every new backend implements them): `from_f32 / from_i32 / to_f32 / to_i32 / full / cast`, `unary`, `binary` (with broadcasting), `expand`, `where_cond`, `reduce`, `matmul` (batched, with transpose flags), `permute`, `narrow / cat`, `index_select / index_add`, `gather / scatter_add`, `im2col / col2im`
- **Fused primitives** (with default implementations built from the required ones; backends override them for speed): `softmax`, `log_softmax`, `layer_norm` (with their backward), `adam_step`

So a new backend trains everything once the required primitives work, and then overrides fused primitives one by one; `tests/backend.rs` compares every fused version with the default composition.

Autograd, all layers and the optimizers work only through `Tensor` operations and never touch memory directly. For example:
- `conv2d` = `im2col` + `matmul` (backward with `col2im`)
- `max_pool2d` = `im2col` + `max` (backward with a mask)
- optimizers compute new weights with Tensor operations under `no_grad` and write them back with `set_storage`, so they run on any device

**Strided views (v2.0)**: a `Tensor` is a buffer plus `(shape, strides, offset)`. `permute` / `transpose` / `t`, `narrow` / `select`, `expand` (stride 0 on broadcast dimensions), `unsqueeze` / `squeeze`, and `reshape` of row-major data copy nothing. Backend primitives still take contiguous buffers: `Tensor::storage()` copies a view once with the `strided_copy` primitive and caches it (data is immutable, so a view is a snapshot and a later `set_storage` does not change it); `matmul` reads transposed and batch-strided views directly (attention's `k.t()`, for example, is not copied on CPU and CUDA; MPS's GEMM has no custom batch strides and copies first). `is_contiguous()`, `contiguous()` and `strided_parts()` inspect and control the layout.

## GPU backends

### MPS (Apple Metal)

- **Memory**: `MTLBuffer`s in unified memory (`StorageModeShared`), so CPU↔GPU transfers are a memcpy; kernel outputs come from a buffer pool sorted by size, so nothing is allocated per op
- **Kernels**: `mps_kernels.metal` is compiled at run time with safe math (no Xcode needed), numerically consistent with the CPU
- **matmul**: `MPSMatrixMultiplication` (batched, transposed), kernel objects cached by shape
- **Asynchronous**: operations accumulate in one command buffer, committed every 48 ops or when data is read back (`to_vec`, `item`), so the CPU never waits on every op
- **Fused primitives**: softmax / log_softmax / layer_norm (with backward) and Adam are single kernels (one threadgroup per row, simdgroup reductions)
- **Half precision**: kernels are templates instantiated for float / half / bf16 (bf16 stored as raw bits, round to nearest even), always computing in f32; F16 matmul uses `MPSMatrixMultiplication` Float16, BF16 computes in f32
- Requires Apple Silicon and macOS 15+ (`atomic_float`, `MTLMathMode`)

### CUDA (NVIDIA)

- **Building**: `cudarc` loads the CUDA driver / NVRTC / cuBLAS at run time, so `--features cuda` compiles on any platform; without a GPU, `Device::Cuda(0).is_available()` is `false`
- **Kernels**: `cuda_kernels.cu` ports `mps_kernels.metal` one to one (warp shuffle reductions, `atomicAdd`), compiled by NVRTC on first use (without fast-math)
- **Half precision**: kernels are templates for float / f16 / bf16 (f16 conversions with inline PTX `cvt`, no `cuda_fp16.h` needed); cuBLAS half GEMMs accumulate in f32 (`CUBLAS_COMPUTE_32F`)
- **matmul**: `cublasSgemmStridedBatched` (half precision through `GemmStridedBatchedEx`), mapping row-major to cuBLAS's column-major by `C = A·B ⇔ Cᵀ = Bᵀ·Aᵀ`; batch stride 0 broadcasts directly
- **Asynchronous**: one stream per GPU, stream-ordered memory (`cuMemAllocAsync`), synchronizing only to read data back; several GPUs as `Device::Cuda(i)`
- **Requires**: an NVIDIA driver and CUDA 12.x's NVRTC and cuBLAS libraries (`libnvrtc` / `libcublas` on the library path; the full toolkit is not needed)

Verify everything on an NVIDIA machine with one command (cross-device comparisons, the full test suite on the GPU, examples, benchmarks):

```bash
./scripts/verify_cuda.sh
```

> ⚠️ The CUDA backend was developed without an NVIDIA GPU: verified are compilation, the kernels' syntax (checked with a C++ shim), kernel names and argument layouts, cuBLAS's row-major argument mapping (simulating column-major semantics in Python, 48 combinations), and the behavior without a GPU; **it has not run on real hardware yet**.

## Performance (Apple M3, 8-core CPU / 8-core GPU, `cargo run --release --example bench <device>`)

| Benchmark | v0.1 CPU | v0.3 CPU | CPU + `accelerate` | MPS (GPU) |
|---|---|---|---|---|
| matmul 1024×1024 | 127 GFLOP/s | 283–380 GFLOP/s | 983 GFLOP/s | 1178 GFLOP/s (about 2000 sustained) |
| softmax (1M elements) | 1.52 ms | 0.43 ms | 0.43 ms | 0.11 ms |
| layer_norm (1M elements) | 1.07 ms | 0.41 ms | 0.44 ms | 0.095 ms |
| conv2d fwd+bwd | 13.3 ms | 6.2 ms | 6.3 ms | 4.4 ms |
| CNN training step (small model) | 10.1 ms | 5.9 ms | 5.1 ms | 5.5 ms |
| GPT training step (d=64, L=2) | 29.1 ms | 9.2 ms | 6.2 ms | 3.4 ms |
| GPT training step (d=256, L=4) | 296 ms | 105 ms | 58 ms | 29 ms |

Mixed precision (the last items of `bench`, Apple M3):

| Benchmark | f32 | f16 | bf16 |
|---|---|---|---|
| matmul 2048×2048 (MPS) | 2145 GFLOP/s | 2206 GFLOP/s | 1903 GFLOP/s (MPS has no bf16 GEMM; computed in f32) |
| GPT training step (d=256, L=4, MPS, autocast) | 28.7 ms | 31.9 ms | 37.0 ms |
| GPT training step (CPU, autocast) | 102 ms | 133 ms | 144 ms |

> On Apple M1–M3, FP16 and FP32 have the same arithmetic throughput, so autocast is **not faster** on this machine (the extra conversions make it about 10% slower); the CPU has no half-precision units and is slower too. Half precision should pay off on NVIDIA Tensor Cores (supported by the CUDA backend, not yet measured); on any device it halves the memory of weights and activations, and loads F16/BF16 weights directly.

Quantized decoding-style matrix-vector products (`[1, 2048] @ [2048, 2048]`, the last item of `bench`, Apple M3):

| | f32 | int8 | int4 |
|---|---|---|---|
| CPU | 0.86 ms | 0.30 ms | 0.29 ms |
| MPS | 0.22–0.33 ms | 0.08 ms | 0.17 ms |

Decoding is bound by memory bandwidth, so smaller weights are faster (MPS's fused kernel splits k over 8 simdgroups and reduces).

Attention forward (`[B=4, H=8, T, D=64]`, causal, the last item of `bench`):

| T | CPU fused | CPU composed | MPS fused | MPS composed | scores of the composed version |
|---|---|---|---|---|---|
| 128 | 0.84 ms | 1.26 ms | 0.86 ms | 0.60 ms | 2 MB |
| 512 | 13.4 ms | 22.4 ms | 6.3 ms | 7.3 ms | 34 MB |
| 1024 | 66.8 ms | 255.9 ms | 36.6 ms | 49.7 ms | 134 MB |

The CPU's fused kernel works on 64×128 blocks (two small GEMMs + an online softmax per block) and never stores the `[Tq, Tk]` scores; MPS / CUDA use blocked GEMMs (at most 16M scores per block), so memory stays bounded for long sequences.

GPU timings measure throughput (N runs, then one synchronization, the `torch.cuda.synchronize()` convention). Very small operations (a 256×256 matmul, a small CNN) are bound by per-op dispatch on the GPU and are not always faster than the CPU; the larger the model, the larger the GPU's advantage.

## Tests

```bash
cargo test                                         # CPU
RUSTNN_DEVICE=mps cargo test --features mps        # the same tests, all on the GPU
cargo test --features mps --test devices           # every backend primitive: GPU against CPU
```

- `tests/gradcheck.rs`: central-difference gradient checks of every differentiable operation (broadcasting, all batch forms of matmul, conv, pooling, softmax, layer norm, attention, ...), and that the causal mask does not look ahead
- `tests/backend.rs`: fused kernels agree with the default compositions, matmul (transposed, split in parallel) against a naive implementation, dtype behavior, values of new operations
- `tests/devices.rs`: cross-device consistency: every primitive (edge sizes, broadcasting, matmul with all transposes / batch strides, conv with all strides / paddings, scatter with repeated indices) must match the CPU on every available GPU backend; also the F16/BF16 versions of all primitives, int8 matmul (both the fused kernel and the GEMM path), int4, and attention
- `tests/io.rs`: files **assembled by hand** from the safetensors specification (not by the code under test) with several dtypes and metadata, byte-by-byte checks of written headers and data, the old format, `load_state_dict` reports; AdamW and SGD-momentum checkpoints resume to **bit for bit identical** parameters (including dropout's RNG); gradient accumulation equals the full batch
- `tests/data.rs`: DataLoader order, shuffles are reproducible permutations, the same result for any number of workers, real parallelism, early exit, worker panics, collate; schedule values; metrics
- `tests/amp.rs`: half-precision values and promotion, differentiable casts, autocast rules, AMP gradients agree with f32 ones (F16 within 2%, BF16 within 8%), GradScaler (unscaling, skipped steps, growth, saved state), F32/F16/BF16 training converges, half-precision CNN inference, safetensors keeps half types
- `tests/inference.rs`: the KV cache's step-by-step / chunked-prefill logits agree with the full forward, generation is the same with and without the cache (also with a sliding window), GPT-2's HF parameter names and checkpoint loading (with the `transformer.` prefix and attention mask buffers), int8 accuracy, BPE (merges with a custom vocabulary, byte-level round trips of any Unicode), the pre-tokenizer regex's edge cases, the sampler's distribution and reproducibility; **with `data/gpt2`** also the golden values of HF's official GPT-2
- `tests/llama.rs`: RMSNorm values, RoPE keeps norms and its dot products depend only on relative positions, `repeat_kv` layout, llama3 frequency scaling, parsing HF's `config.json` and round trips, with GQA the KV cache's step-by-step / chunked prefill agrees with the full forward, causality and batch consistency, the same generation with and without the cache, HF layout (`[out, in]` shapes, element-wise transposes) saving / loading, learning sequences, int8, autocast gradients agree with f32
- `tests/vision.rs`: ResNet-18/34/50 and ViT-B/16 parameter counts exactly as torchvision / timm, ResNet parameter names and shapes, torchvision layout (1-d BN, `[out, in]` fc, `num_batches_tracked`) loads to the same outputs, max pooling pads with -inf, small ResNet / ViT learn to classify
- `tests/seq2seq.rs`: the decoder's KV cache agrees with teacher forcing, with a padding mask the source padding does not change the output at all (without one it does), `cross_entropy_ignore`, learning to reverse sequences
- `tests/finetune.rs`: LoRA starts as the identity, only adapters train, frozen parameters do not change, outputs are the same before and after merging, saving and loading only the adapters, training on an int8 base; gradient checkpointing gives the same gradients on Llama / Llama + LoRA / GPT with dropout / GPT-2
- `tests/quant.rs`: int4 packing and dequantization agree element-wise with an independent computation, fused kernels agree with the default (dequantize on the host + GEMM), int4 error on Llama / GPT-2, LoRA trains on an int4 base
- `tests/attention.rs`: fused attention agrees with the composition under all masks (none, causal, padding, per head, full, single row), `dv ≠ d`, half precision, autocast; the same gradients, the same results for any block size, numerically stable with long sequences and large logits
- `tests/views.rs` / `tests/kvcache.rs`: views copy nothing and agree with host computations, views are snapshots, matmul reads transposed and strided views directly, all dtypes; the KV cache doubles its capacity, returns views, earlier views and clones are not affected by later writes, gradients flow through the cache, 100 decoding steps agree with the full forward
- `tests/second_order.rs`: Hessian-vector products of element-wise ops, all matmuls, softmax / log_softmax / cross-entropy, layer / RMS norm, embedding + gather, conv2d, fused attention and gradient checkpointing agree with finite differences; the semantics of `backward_create_graph` and `autograd::grad`; gradient penalties reach the parameters with the gradients of numerical differentiation
- `tests/world.rs`: 6 environments × 4 seeds × 400 steps (with autoreset) compared step by step, bit for bit, with Gymnasium; vector environments, FrozenLake's text rendering, NumPy random numbers (seeds beyond 2⁶³, all `integers` ranges)
- `tests/agent.rs`: `ruai agent`'s embedded translation is up to date; about 50 offline commands (the echo model) with interactive chats, and the output against a fake LLM server (`tests/agent/fake_llm.py`: OpenAI / Anthropic, streaming or not, native tool calls, HTTP 500 retries) are identical to CPython running the same py0 sources
- `tests/py0.rs` + `tests/py0/`: py2rs's corpus, 259 programs (with `tests/py0/numpy/`, `sklearn/`, `gymnasium/`, `torch/` `plot/` and the Python bridge's `pybridge/` compared with the real NumPy, scikit-learn, Gymnasium and PyTorch; each program runs on a Python that has its packages), each translated, compiled and run, with the output, exit code and exception type of CPython 3.12+; 46 programs that must be rejected, with the expected messages; `tests/py0/pyo3/run.sh` checks the PyO3 extension modules
- `tests/training.rs`: an MLP learns XOR, a CNN (with BatchNorm) converges, a GPT memorizes sequences, save/load round trips

## Stability

From 4.0 ruai follows semantic versioning for:

- the Rust API of `ruai::nn` (also re-exported at the crate root), `ruai::world` and `ruai::rng`;
- the `ruai` command line (subcommands and their options);
- the py0 language ([py0/SPEC.md](py0/SPEC.md)) and the behavior of its packages: a program that runs on one 4.x runs, with the same output, on later 4.x versions.

`ruai::py` (the translator) and `ruai::pyrt` (the runtime that translated programs call) are public because generated code uses them, but their Rust API follows the translator and may change in minor versions; translated programs depend on the matching ruai version.

## Versions

| Version | Contents |
|---|---|
| v0.1 | CPU backend, autograd, CNN, Transformer/GPT, optimizers, saving |
| v0.2 | DType (I32 indices), new operations (cat/gather/where, ...), fused primitives, fast GEMM, Accelerate |
| v0.3 | MPS (Apple Metal) backend, cross-device test framework, `--device` |
| v0.4 | CUDA backend (cuBLAS + NVRTC), several GPUs (not yet run on hardware) |
| v0.5 | DataLoader, full checkpoints (optimizer state and RNG), safetensors, LR schedules, gradient accumulation, metrics |
| v0.6 | f16 / bf16 (CPU / MPS / CUDA), autocast, GradScaler, half-precision inference |
| v0.7 | KV cache, loading GPT-2's official weights, BPE tokenizer, sampling, int8 quantization |
| v0.8 | RMSNorm, RoPE, SwiGLU, GQA, the Llama model (HF layout import / export) |
| v0.9 | ResNet, ViT, encoder-decoder Transformer (cross-attention, seq2seq) |
| v1.0 | API cleanup (`GPT::generate` takes a `Sampler`), rustdoc for every public item, clean clippy, `cargo package` |
| v1.1 | GitHub Actions CI (Linux / macOS, with a manual GPT-2 real-weights job), `zeros_like` keeps the dtype, `collator` keeps image shapes |
| v1.2 | Hugging Face `tokenizer.json` tokenizers (`HfTokenizer`), `llama --model` loads pretrained models |
| v1.3 | LoRA (also on int8 bases), gradient checkpointing, `freeze` / `trainable_parameters` |
| v1.4 | grouped int4 weight quantization (fused CPU / MPS / CUDA kernels, LoRA on top) |
| v1.5 | fused attention (CPU: FlashAttention-style blocks; other backends: blocked GEMM with bounded memory), faster MPS quantized matvec |
| v2.0 | strided views without copies, a preallocated KV cache (decoding about 1.4x faster on CPU, 6x on MPS), second derivatives (`create_graph`, `autograd::grad`) |
| v3.0 | one crate with features, the engine in `src/nn/`, the git-style `ruai` command (`ruai nn devices / info / generate`, external `ruai-NAME`) |
| v3.1 | `world`: Gymnasium 1.x's API and environments (bit for bit), NumPy-compatible `ruai::rng`, `ruai world list / run`, a REINFORCE example |
| v3.2 | `py`: py2rs merged with `git subtree` (translator `ruai::py`, runtime `ruai::pyrt`), `ruai py run / emit / check / explain / build / pyo3 / stub / test / fuzz` |
| v3.3 | py0 packages (`import M [as A]` / `from M import x` → `src/py/lib/M.pyi` + `ruai::pyrt::lib::M`), Python's standard library in Rust (first: `shlex`, `fnmatch`, `html`) |
| v3.4 | `ruai agent`: agge rewritten in py0 and built into `ruai`; py0's `os` / `sys` / `time` additions, `subprocess`, `ruai.json`, `ruai.http`, `Result[T, OSError]` |
| v3.5 | py0's `numpy` (ndarray, broadcasting, indexing and views, reductions, `linalg`, `random`, printing identical to NumPy's); union parameters and overloads in stubs |
| v3.6 | py0's `sklearn` (linear models, logistic regression, k-NN, naive Bayes, KMeans, PCA, preprocessing, `train_test_split`, metrics, iris) |
| v3.7 | py0's `gymnasium` (world's environments, bit for bit as Gymnasium) and `torch` (autograd on the nn engine, `nn` modules, losses, `optim`, PyTorch's random numbers for `manual_seed`); float32 numpy arrays |
| v4.0 | stable release: semantic versioning (see Stability), English documentation |

## Limitations

- Half precision on the CPU is a storage format (computed in f32) and is not faster
- `index_add` / `scatter_add` / fused kernels have only f32 versions on GPUs; half-precision inputs are converted to f32 (autocast runs them in f32 anyway)
- Second derivatives through fused attention and gradient checkpointing recompute the forward once
- Attention on GPUs is blocked GEMMs, not one fused kernel (a real GPU FlashAttention needs simdgroup matrix tiles); backward recomputes and builds the scores temporarily
- MPS: small operations are bound by dispatch overhead
- The CUDA backend has not run on an NVIDIA GPU yet (see above and `scripts/verify_cuda.sh`)
- `HfTokenizer` supports only BPE models; Llama 3's `rope_scaling` and Llama 2-style (SentencePiece) `tokenizer.json` are not yet verified with real files
- MPS / CUDA: at most 8 tensor dimensions; element counts must fit in u32
