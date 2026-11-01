# ruai（rust-nn）

[![CI](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml/badge.svg)](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml)

[English](README.md) | 繁體中文

用 Rust 寫的 AI 瑞士刀：一個 crate、一個 `ruai` 指令，各部分是 cargo feature，像 git 一樣用子指令呼叫。

| 部分 | 指令 | 內容 | 狀態 |
|---|---|---|---|
| `ruai::nn`（feature `nn`） | `ruai nn ...` | 類似 PyTorch 的神經網路引擎：動態計算圖 + 自動微分（含二階）、CPU / Apple GPU（MPS）/ NVIDIA GPU（CUDA）、CNN、ResNet、ViT、GPT / GPT-2 / Llama、量化、safetensors | ✅ |
| `ruai::world`（feature `world`） | `ruai world ...` | gymnasium 1.x 的 API 與環境（CartPole、MountainCar、Pendulum、Acrobot、FrozenLake），相同 seed 下與 gymnasium 逐位元相同 | ✅ |
| `ruai::py`（feature `py`） | `ruai py ...` | py0：受限、強型別的 Python，翻成可讀的 Rust 再編譯執行（由 py2rs 併入，見 [py0/](py0/README.tw.md)），附以 Rust 重寫的 Python 標準函式庫與 numpy、scikit-learn、Gymnasium、PyTorch 子集 | ✅ |
| agent（feature `agent`） | `ruai agent ...` | 以 py0 寫成的 agge：LLM agent 團隊寫程式與寫書、graph 工作流程（原始碼 [src/agent/](src/agent/README.md)，CPython 也能執行） | ✅ |

```bash
cargo install ruai                    # 或在 checkout 中：cargo build --release
ruai nn devices                       # 可用的運算裝置
ruai nn info data/smollm2-135m        # 模型摘要（只讀 safetensors 標頭）
ruai nn generate --model data/gpt2 --prompt "The dog" -n 18
ruai nn generate --model data/smollm2-135m --device mps --int8 --prompt "Gravity is"
ruai py run prog.py                   # py0：翻成 Rust、編譯（有快取）並執行
ruai py emit prog.py                  # 印出翻譯的 Rust 程式
ruai world list                       # 內建環境與其空間
ruai world run CartPole-v1 --episodes 3 --seed 1
ruai agent init && ruai agent chat    # LLM agent 團隊：互動對話（/help 看指令）
ruai agent code "寫一個 todo CLI" -o todo   # 設計 → 實作 → 測試 → 審查
ruai NAME ...                         # 其他名稱會執行 PATH 上的 ruai-NAME（像 git）
```

函式庫裡引擎在 `ruai::nn`，也 re-export 在根目錄（`ruai::Tensor` 等同 `ruai::nn::Tensor`，`ruai::nn::Linear`、`ruai::nn::llama::Llama` 等 v2 路徑照舊可用）。只要部分功能時可關掉預設 features：`ruai = { version = "4", default-features = false, features = [...] }`。

## 安裝

```toml
[dependencies]
ruai = "4"     # 或 git 版本：ruai = { git = "https://github.com/ccc-llm/ruai" }
# GPU：features = ["mps"]（Apple）或 ["cuda"]（NVIDIA）；macOS CPU 加速：["accelerate"]
```

> crate 名稱為 [`ruai`](https://crates.io/crates/ruai)（原 `rust-nn` / `rustnn` 在 crates.io 上已被使用），程式中以 `use ruai::...` 引用。各版變更見 [CHANGELOG.md](CHANGELOG.md)。

## 快速開始

```bash
cargo test                              # 梯度數值檢查 + 端到端訓練測試
cargo run --release --example cnn       # CNN 影像分類（有 MNIST 就用 MNIST），每個 epoch 存 checkpoint
cargo run --release --example cnn -- --resume          # 從 runs/cnn/last.safetensors 接續訓練
cargo run --release --example cnn -- --model resnet    # 小型 ResNet（cnn / resnet / vit）
cargo run --release --example cnn -- --model vit       # Vision Transformer（4×4 patch）
cargo run --release --example seq2seq   # encoder-decoder Transformer 學會排序數字串（padding mask + cached greedy）
cargo run --release --example gpt       # 字元級 GPT：訓練 → 存檔 → 載入 → 生成
cargo run --release --example gpt -- --accum 4         # 梯度累積（每步 4 個 micro-batch）
cargo run --release --example bench     # 效能基準測試

# macOS：用 Apple Accelerate (AMX) 做矩陣乘法，大型模型約快 2 倍
cargo run --release --features accelerate --example gpt

# Apple GPU（Metal）：加上 --features mps，用 --device 選裝置（cpu / mps / auto）
cargo run --release --features mps --example gpt -- --device mps
cargo run --release --features mps --example bench mps

# NVIDIA GPU（CUDA）：加上 --features cuda
cargo run --release --features cuda --example gpt -- --device cuda

# 預訓練 GPT-2（Hugging Face 權重）：下載一次後即可生成文字
./scripts/download_gpt2.sh
cargo run --release --example gpt2 -- --prompt "The meaning of life is"
cargo run --release --example gpt2 -- --int8                  # int8 權重
cargo run --release --example llama     # 字元級 Llama（RMSNorm/RoPE/SwiGLU/GQA）→ 匯出 HF 格式 → 重新載入 → 生成
cargo run --release --features mps --example gpt2 -- --device mps

# 混合精度訓練（autocast + f32 主權重；f16 會自動做 loss scaling）
cargo run --release --features mps --example gpt -- --device mps --amp f16
```

程式裡選裝置：`Tensor::randn(&[2, 3], Device::Mps)`、`model.to_device(Device::Mps)`、`Device::best_available()`，或用環境變數 `RUSTNN_DEVICE=mps`。

MNIST：把解壓後的 `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte` 放到 `data/mnist/`；找不到時 `cnn` 範例會改用自動產生的幾何圖形資料集。

## 使用範例

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
// 每個 batch：
opt.zero_grad();
let loss = F::cross_entropy(&model.forward(&x), &labels, 0.0);
loss.backward();
opt.step();

// 推理
model.eval();
let pred = no_grad(|| model.forward(&x).argmax(-1, false));

// 存檔 / 載入（safetensors，可與 PyTorch / Hugging Face 互通）
ruai::serialize::save("model.safetensors", &model)?;
ruai::serialize::load("model.safetensors", &model)?;
```

### 資料載入、排程與 checkpoint

```rust
use ruai::data::{ClassificationData, DataLoader};
use ruai::optim::lr::{CosineLr, LrScheduler};
use ruai::checkpoint::{self, TrainState};

let collate = train_data.collator(dev);                 // [B, C, H, W]（collate 則攤平成 [B, C*H*W]）
let loader = DataLoader::new(train_data, 64, collate)
    .shuffle(true).drop_last(true).num_workers(2);    // 背景執行緒預先組 batch
let sched = CosineLr { base_lr: 1e-3, min_lr: 1e-4, warmup: 100, total: 10_000 };
let mut state = TrainState::default();               // 或 checkpoint::load(...) 接續

for epoch in state.epoch..epochs {
    for batch in loader.epoch(epoch) {                // 每個 epoch 的打亂順序是確定的
        sched.apply(&mut opt, state.step as usize);
        opt.zero_grad();
        F::cross_entropy(&model.forward(&batch.inputs), &batch.targets, 0.0).backward();
        opt.step();
        state.step += 1;
    }
    state.epoch += 1;
    checkpoint::save("runs/last.safetensors", &model, &opt, &state)?;  // 模型 + 優化器 + RNG
}
```

### 預訓練 GPT-2

```rust
use ruai::nn::gpt2::GPT2;
use ruai::tokenizer::Gpt2Tokenizer;
use ruai::generation::Sampler;

let tok = Gpt2Tokenizer::from_dir("data/gpt2")?;
let mut model = GPT2::from_pretrained("data/gpt2", Device::Cpu)?;  // 直接讀 HF 的 model.safetensors
model.quantize_int8();                                              // 可選：int8 權重
let mut sampler = Sampler::new(0.8, 40, 0.95, 42);                  // temperature, top-k, top-p, seed
let ids = model.generate(&tok.encode("Once upon a time"), 50, &mut sampler);  // 使用 KV cache
println!("{}", tok.decode(&ids));
```

> ✅ 已以官方權重驗證（`./scripts/download_gpt2.sh` 後 `cargo test --release --test inference`）：tokenizer 黃金值、greedy 生成 "The dog" 的 20 個 token 與 Hugging Face transformers 的 GPT-2 整合測試（`test_lm_generate_gpt2`）完全相同（CPU 與 MPS）；int8 的 logits 誤差約為 logit 範圍的 1.8%。實測（Apple M3，100 個 token）：CPU f32 / int8 / int4 = 41 / 62 / 88 tokens/s，MPS = 58 / 163 / 178 tokens/s。

### Llama（RMSNorm / RoPE / SwiGLU / GQA）

```rust
use ruai::nn::llama::{Llama, LlamaConfig};

let cfg = LlamaConfig::new(vocab, 256, 4, 8, 2, 512);   // vocab, hidden, layers, heads, kv_heads, max_len
let model = Llama::new(cfg, Device::Cpu);               // 8 個 query head 共用 2 組 K/V（GQA）
let logits = model.forward_tokens(&tokens);             // [B, T] -> [B, T, vocab]
model.save_pretrained("runs/llama")?;                    // config.json + model.safetensors（HF 版面）
let model = Llama::from_pretrained("data/tinyllama", Device::Cpu)?;  // HF LlamaForCausalLM 權重
let ids = model.generate(&prompt_ids, 50, &mut Sampler::greedy());   // KV cache（只快取 kv_heads 份）
```

元件也可單獨使用：`F::rms_norm`、`F::rope_inv_freq` / `F::rope_cos_sin` / `F::apply_rope`（HF 的 rotate_half 版面）、`F::repeat_kv`、`Tensor::silu`、`nn::RMSNorm`、`nn::SwiGLU`。`config.json` 支援 `num_key_value_heads`、`head_dim`、`rope_theta`、`tie_word_embeddings`、`attention_bias` 與 Llama 3.1 的 `rope_scaling`（`rope_type: "llama3"`；其他類型會回報錯誤）。HF 的 `nn.Linear` 權重為 `[out, in]`，本專案 `Linear` 為 `[in, out]`，`from_pretrained` / `save_pretrained` 會自動轉置。

預訓練模型（`config.json` + `model.safetensors` + `tokenizer.json`）：

```bash
./scripts/download_smollm.sh                    # HuggingFaceTB/SmolLM2-135M → data/smollm2-135m
cargo run --release --example llama -- --model data/smollm2-135m --prompt "The capital of France is"
```

```rust
use ruai::tokenizer::HfTokenizer;
let tok = HfTokenizer::from_dir("data/smollm2-135m")?;          // 讀 tokenizer.json
let ids = tok.encode("Hello world", true);                      // true：加上 BOS 等特殊 token
println!("{}", tok.decode(&ids, true));                         // true：略過特殊 token
```

`HfTokenizer` 支援 BPE 模型（GPT-2、Llama 2/3、TinyLlama、SmolLM、Qwen …）的 normalizer、pre-tokenizer（ByteLevel、含 lookahead 的 Split regex、Metaspace、Digits …）、byte fallback、added / special token、post-processor 範本與 decoder；不支援的元件（WordPiece、Unigram 等）載入時會回報錯誤。

> ✅ 已以 SmolLM2-135M 的官方權重驗證（`./scripts/download_smollm.sh` 後 `cargo test --release --test llama --test tokenizer`）：`tokenizer.json` 分詞結果正確，logits 與另外以純 PyTorch 依 Hugging Face `LlamaForCausalLM` 寫成的獨立實作相差不到 3e-4（每個位置的 argmax 都相同，CPU 與 MPS）；GPT-2 的 `tokenizer.json` 與 `vocab.json` + `merges.txt` 逐 token 相同。實測（Apple M3，40 個 token）：CPU f32 / int8 / int4 = 36 / 53 / 61 tokens/s，MPS = 29 / 92 / 99 tokens/s（30 層的小模型在 MPS 上受每個 op 的派送開銷限制）。尚未驗證：Llama 3 的 `rope_scaling`、Llama 2 的 SentencePiece 式 `tokenizer.json`。

### py0：強型別的 Python 子集（py）

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

**py0 套件與標準函式庫**：`import M` / `import M as A` / `from M import x` 會在 `src/py/lib/M.pyi`（以及 `PY0_LIB` 列出的目錄）找套件 stub——以型別宣告描述函式與類別，對應到 `ruai::pyrt::lib::M` 下的 Rust 實作（或 stub 開頭 `# rust:` 指定的路徑；需要額外依賴時以 `# crate: pyrt/FEATURE` 開啟 ruai 的執行期 feature）。Python 標準函式庫逐模組以 Rust 重寫在 `src/pyrt/lib/`，每個模組都有與 CPython 對照的測試程式（`tests/py0/py/std_*.py`）。目前有 `shlex`（split / quote / join）、`fnmatch`（fnmatch / fnmatchcase / filter）、`html`（escape / unescape，含完整 HTML5 實體表）、`subprocess.run`，以及內建模組的擴充：`os`（getcwd、environ.get、listdir、makedirs、rmdir、rename、getpid）、`os.path`（isdir、isfile、abspath、realpath、expanduser、dirname、basename、splitext、relpath、normpath、getsize）、`sys.stdin.isatty` / `sys.stderr.write`、`time.strftime`；另外 py2rs 原本內建 `math`、`collections`、`heapq`、`bisect`、`itertools`、`functools`、`random`、`re`、`sys`、`time`、`json.dumps`、`os.path`。stub 可宣告 `Result[T, OSError | ...]`，讓 Rust 實作丟出 CPython 的例外類別。ruai 自己的套件：`ruai.json`（`Json`：有靜態型別的 JSON 值，`dumps` 與 CPython 的 `json.dumps` 逐字相同）、`ruai.http`（HTTP(S) 請求，逐行讀取回應以支援 server-sent events，feature `pyrt-http`）；它們在 `py0/python/ruai/` 有給 CPython 用的對應實作，所以同一支程式兩邊都能跑。

`ruai py run prog.py` 把它翻成可讀的 Rust（`ruai py emit` 可看），以 Cargo 編譯（第一次需要幾秒，之後有快取）後執行，輸出與 CPython 相同；做不到時在翻譯階段就以行號與提示拒絕。py0 由 [py2rs](py0/README.tw.md) 併入（保留完整 git 歷史），規格見 [py0/SPEC.md](py0/SPEC.md)。翻譯出的程式依賴 `ruai`，但只開執行期 feature（`default-features = false, features = ["pyrt"]`），不會編譯神經網路引擎。執行 py0 程式需要 Rust 工具鏈（cargo）。

### LLM agent 團隊（agent）

```bash
ruai agent init                                   # 在目前目錄建立 .agge/（設定、agent、graph、紀錄）
ruai agent model add glm nvidia z-ai/glm-5.3-flash   # provider：openai、nvidia、openrouter、deepseek、groq、gemini、ollama、lmstudio、anthropic
ruai agent chat --team code                       # 與 lead 對話，lead 以 <delegate> 分派給 architect / coder / tester / reviewer
ruai agent ask -y "建立 hello.py 並執行"            # 單次任務：agent 以文字標籤呼叫工具（讀寫檔、搜尋、shell）
ruai agent book "Git 入門" --chapters 3 -o gitBook   # 大綱 → 各章（map 節點）→ 總編輯
ruai agent run brainstorm "如何設計快取層" -M merge=claude
```

`ruai agent` 是 [agge](src/agent/README.md)（約 2,500 行的 Python）以 py0 改寫後翻成 Rust、編進 `ruai` 執行檔，不需要 Python 或 cargo。同一份原始碼（`src/agent/*.py`）也能直接以 CPython 執行；`tests/agent.rs` 用同一串指令（含一個假的 OpenAI / Anthropic 串流伺服器）比對兩者的輸出逐字相同，並檢查嵌入的翻譯 `src/cli/agent_gen.rs` 是最新的。與原版的差異：平行的 delegate、graph 節點與 map 項目改為依序執行；聊天介面沒有 readline 歷史與補完。

### py0 的 numpy

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

`import numpy as np` 在 py0 中是以 Rust 實作的 NumPy 子集（`src/pyrt/lib/numpy/`）：float64 / int64 / bool 陣列、NumPy 2 的型別提升與廣播、切片是共用記憶體的 view、布林遮罩與索引陣列、沿軸的歸約（`sum` 採用 NumPy 的 pairwise 加總順序）、`linalg`、與 NumPy 逐位元相同的 `random.default_rng`；`print` / `repr` 與 NumPy 的 `arrayprint` 逐字相同。同一支程式用 CPython 執行時就是真的 NumPy，`tests/py0/numpy/` 逐支比對。因為 py0 是靜態型別：`a[i]` 與歸約的結果是 0 維陣列（NumPy scalar），需要 Python float 時寫 `float(a.sum())`；`a.shape` 可索引但不能拆包；dtype 用字串（`dtype="int64"`）。浮點矩陣乘法依序加總，可能與 NumPy 的 BLAS 在最後一位不同。

### py0 的 sklearn

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

`from sklearn.X import Y` 在 py0 中是以 Rust 實作、建在 numpy 陣列上的 scikit-learn 子集（`src/pyrt/lib/sklearn/`）：`LinearRegression`、`Ridge`、`LogisticRegression`、`KNeighborsClassifier` / `Regressor`、`GaussianNB`、`KMeans`、`PCA`、`StandardScaler`、`MinMaxScaler`、`train_test_split`、常用指標與 `load_iris`。演算法依照 scikit-learn（相同的置中、求解器的最佳解、PCA 的正負號慣例、`train_test_split` 用同樣的 `RandomState` 亂數，所以切分完全相同），`tests/py0/sklearn/` 以真的 scikit-learn 對照輸出。差異：scikit-learn 提早停止的迭代求解器（lbfgs）只在其容許誤差內相同；k-NN 距離剛好相等時可能選到不同的鄰居；`KMeans` 需要給初始中心（`init=陣列`）；`target_names` 是 list。

### py0 的 gymnasium 與 torch

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

`import gymnasium as gym` 是 `ruai::world` 的環境套上 Gymnasium 的 Python API（觀測值是 float32 的 numpy 陣列，與 Gymnasium 逐位元相同）；`import torch` 是建在 ruai nn 引擎上的 PyTorch 子集：autograd、`nn.Linear` 等模組、`nn.Sequential`、損失函數、`optim.SGD` / `Adam`、`torch.no_grad()`。`torch.manual_seed` 重現 PyTorch 的亂數產生器（MT19937），所以 `rand`、`randn` 與 `nn.Linear` 的初始權重都與 PyTorch 相同；張量與模組的印出格式也一樣。`tests/py0/gymnasium/`、`tests/py0/torch/` 以真的 Gymnasium 與 PyTorch 對照（訓練迴圈在 PyTorch 預設的 4 位小數下相同）。差異：模型用 `nn.Sequential` 或持有各層的一般類別（不能繼承 `nn.Module`），`parameters()` 回傳 list；dtype 在 py0 中是字串；矩陣乘積與元素函數可能與 PyTorch 的 BLAS / 向量化實作在最後一位不同。

### 繪圖：ruai.plot

```python
import numpy as np
import ruai.plot as plt

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), "--", label="cos")
ax.legend()
plt.show()          # 在瀏覽器中開啟，畫在 <canvas> 上
```

`ruai.plot`（Rust 中為 `ruai::plot`，feature `plot`，無相依套件）提供 matplotlib.pyplot 的 API：`plot`（格式字串如 `"r--o"`）、`scatter`、`bar`（含類別軸）、`hist`、`imshow`、`text`、`axhline` / `axvline`、標題、軸標籤、範圍、圖例、格線、`subplots` / `subplot`，以及 `Figure` / `Axes` / `Line2D` 物件。自動縮放與刻度採用 matplotlib 的規則（5% 邊界、sticky edges、`MaxNLocator`），所以 `get_xlim()` / `get_xticks()` 與 matplotlib 數值相同，`tests/py0/plot/` 逐一對照。`plt.show()` 產生含 canvas 的 HTML 頁面並以瀏覽器開啟（`MPLBACKEND=agg` 時不開啟）；`savefig` 輸出 `.svg` 或 `.html`。同一程式在 CPython 下直接使用 matplotlib（`py0/python/ruai/plot.py`）。尚未支援：PNG 輸出、`subplots(n, m)` 的二維格子（請用 `subplot`）、colorbar、對數座標。

### py0 → WebAssembly，在瀏覽器中編譯

```bash
ruai py wasm prog.py            # prog.wasm + ruai_wasm.mjs（宿主程式）
node ruai_wasm.mjs prog.wasm    # 或：ruai py wasm --run prog.py
sh py0/web/build.sh && python3 -m http.server -d py0/web 8000   # playground
```

py0 也能**不經過 Rust，直接編譯成 WebAssembly**：`ruai::py::wasm` 把型別化的 IR 轉成 WasmGC 模組（物件、list、dict、set、tuple 與字串都是 GC 的 struct 與 array；例外使用 wasm 的例外處理），執行期函式直接生成在模組內，再加上一個小小的 JavaScript 宿主（`ruai_wasm.mjs`）負責輸出、浮點數的格式化與解析、數學函式和 str 方法，行為與 CPython 一致（最短的浮點 repr、正確捨入的格式規格）。翻譯器本身可以編譯成 `wasm32-unknown-unknown`，所以 `py0/web/` 的 **playground** 在瀏覽器裡把 py0 編成 wasm（每個程式幾毫秒）並執行，不需要伺服器或工具鏈。`tests/py0/` 的 263 支程式中有 221 支能編成 wasm，且輸出全部與 CPython 相同（`ruai py test --wasm`）；尚未支援：generator、函式當值、`nonlocal`、檔案、`re`、`heapq` / `itertools` / `random`，以及以 Rust 實作的套件（numpy、sklearn、gymnasium、torch、ruai.plot）。

### 網站：在瀏覽器中執行 ruai

```bash
./run.sh dev       # 建置 site/ 並在 http://localhost:8000 提供（./run.sh help：所有指令）
```

`site/` 是靜態網站，所有功能都編成 WebAssembly，由 `.github/workflows/pages.yml` 發佈到 GitHub Pages。網站裡有 VS Code 風格的 **py0 IDE**（檔案總管，檔案存在瀏覽器裡；分頁、Monaco 編輯器、終端機；編譯錯誤直接標在程式碼上；標準輸入；可以中斷無限迴圈的停止鍵），以及 Jupyter 風格的 **py0 notebook**（程式與 Markdown 格子，最後一行運算式顯示 `Out[n]`，可以匯出與開啟 `.ipynb`）。另外還有這些展示：MLP 分類器 playground、曲線擬合、SGD / momentum / Adam 走 Rosenbrock 山谷、ConvNetJS 風格的 MNIST CNN（各層 activation 視覺化、可以手寫數字讓它辨識）、用 WebGPU 訓練 GPT、REINFORCE 學 CartPole、Q-learning 學 MountainCar、value iteration 解 FrozenLake、用鍵盤玩 classic-control 環境，以及 `ruai.plot` 圖庫。`./run.sh` 也提供 `setup`、`check`、`build`、`serve`、`watch`、`test`、`size`、`clean`。建置需要 cargo 的 `wasm32-unknown-unknown` target 與 `wasm-bindgen-cli` 0.2.130，不需要任何 npm 套件。詳見 [doc/site.md](doc/site.md)。

### 任何 Python 函式庫：Python 橋接

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6.28, 50)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(x, np.sin(x), "r-", label="sin")
ax.legend()
fig.savefig("sin.png")
```

除了以 Rust 重寫的套件，py0 也能呼叫**真正的 Python 函式庫**：第一行是 `# python: MODULE` 的 stub 用 py0 型別宣告該模組的函式與類別，每次呼叫都交給嵌入程式中的 CPython（PyO3，執行期 feature `pyrt-python`）。參數與結果雙向轉換（int、float、bool、str、None、list、dict、tuple、numpy 陣列）；預設值寫 `= ...` 表示交給 Python 自己的預設值；Python 的例外變成同名的 py0 例外。`matplotlib` 與 `matplotlib.pyplot` 附有整理好的 stub。其他模組可用 `ruai py stub --python MODULE` 以 `inspect` 產生草稿到 `py0lib/MODULE.pyi`（在程式旁、`./py0lib` 與 `PY0_LIB` 中尋找）：有標註的型別對應到 py0 型別，未知的成為 `Object`（任何 Python 物件，可接受任何值），再手動收緊型別。程式連結到第一個能 import 這些模組的 Python（`PY0_PYTHON`，否則 python3.14 / 3.13 / 3.12 / python3），執行時使用該直譯器的套件。取捨：執行時需要安裝那個 Python，呼叫是 CPython 的速度，型別檢查只到 stub 為止。`tests/py0/pybridge/` 以 CPython 對照橋接的程式（matplotlib，以及作為範例的標準模組）。

### 強化學習環境（world）

```rust
use ruai::world::{self, Value};

let mut env = world::make("CartPole-v1")?;            // 同 gymnasium.make：含 TimeLimit(500)
env.action_space_mut().seed(1);
let (mut obs, _) = env.reset(Some(42));               // 同 env.reset(seed=42)
loop {
    let a = env.action_space_mut().sample();          // Value::Int / Value::Floats
    let s = env.step(&a);                             // obs, reward, terminated, truncated, info
    if s.terminated || s.truncated { break }
    obs = s.obs;
}
```

環境的動力學、浮點運算順序與亂數（`ruai::rng`：NumPy 的 `SeedSequence` + `PCG64`）都與 gymnasium 1.x 相同，所以相同 seed 會得到逐位元相同的觀測值、獎勵與結束旗標（`tests/world.rs` 在裝有 gymnasium 時逐步比對）。另有 `RecordEpisodeStatistics`、`SyncVectorEnv`（gymnasium 1.x 的 next-step autoreset）。`cargo run --release --example cartpole` 用 `ruai::nn` 的策略網路以 REINFORCE 解 CartPole（約 40 次更新）。

### 微調：LoRA 與 gradient checkpointing

```rust
let mut model = Llama::from_pretrained("data/smollm2-135m", dev)?;
model.add_lora(8, 16.0, &["q_proj", "v_proj"]);       // 凍結整個模型，只訓練低秩 adapter（rank 8，alpha 16）
model.set_gradient_checkpointing(true);              // backward 時重算每層 activation，省記憶體
let mut opt = Adam::adamw(model.trainable_parameters(), 1e-4, 0.0);
// ...訓練...
serialize::save_tensors("adapter.safetensors", &nn::lora_state_dict(&model))?;  // 只存 adapter（很小）
model.merge_lora();                                   // 併回權重後可 save_pretrained 匯出成 HF 格式
```

`add_lora` 也可用在 `quantize_int8()` 之後（QLoRA 式：int8 底模 + f32 adapter）。`F::grad_checkpoint(inputs, params, f)` 可用在任意子網路，重算時會還原 RNG 狀態與 autocast 模式，所以含 dropout 的梯度也與不重算時完全相同。

### ResNet / ViT / encoder-decoder

```rust
use ruai::nn::resnet::{ResNet, ResNetConfig};
use ruai::nn::vit::{ViT, ViTConfig};
use ruai::nn::seq2seq::{Seq2Seq, Seq2SeqConfig};

let resnet = ResNet::new(ResNetConfig::resnet50(1000), dev);         // 參數數量與 torchvision 完全相同
resnet.load_torchvision("resnet50.safetensors")?;                    // torchvision state_dict（存成 safetensors）
let small = ResNet::new(ResNetConfig::small(3, &[16, 32, 64], 2, 10), dev);  // CIFAR 式 stem
let vit = ViT::new(ViTConfig::base_16(1000), dev);                   // ViT-B/16：86,567,656 個參數（同 timm）

let s2s = Seq2Seq::new(cfg, dev);
let mask = F::padding_mask(&src, PAD);                               // [B, 1, 1, Ts]
let logits = s2s.forward_pair(&src, &tgt_in, Some(&mask));           // teacher forcing
let loss = F::cross_entropy_ignore(&logits.reshape(&[b * t, v]), &tgt_out, PAD);
let out = s2s.greedy(&src_ids, BOS, Some(EOS), 50);                  // 解碼端 self-attention 用 KV cache
```

### 二階導數

```rust
use ruai::autograd;

// gradient penalty（WGAN-GP）：對輸入的梯度再對參數微分
let x = x.detach().requires_grad_();
let gx = autograd::grad(&d.forward(&x).sum_all(), &[&x], true)[0].clone().unwrap();  // create_graph = true
let norm = (&gx * &gx).sum(1, false).sqrt();
let penalty = (norm.add_scalar(-1.0).powf(2.0)).mean_all();
penalty.backward();                                   // 梯度傳到 d 的參數

// 或像 PyTorch 的 backward(create_graph=True)
loss.backward_create_graph();                         // 之後 x.grad() 本身可再微分
```

每個運算的反向函式在 `create_graph` 時改用可微分的運算（平常仍走融合 kernel），`tests/second_order.rs` 以有限差分檢查各運算的 Hessian-向量積。

### 混合精度（AMP）

```rust
use ruai::amp::{autocast, GradScaler};

let mut scaler = GradScaler::new();                   // f16 需要；bf16 可省略
opt.zero_grad();
let loss = autocast(DType::F16, || F::cross_entropy(&model.forward(&x), &y, 0.0));
scaler.scale(&loss).backward();                       // 參數維持 f32，收到 f32 梯度
scaler.unscale(&opt);                                 // （要做 clip_grad_norm 時）
scaler.step(&mut opt);                                // 梯度溢位時自動跳過並調降 scale

// 純半精度推理
model.to_dtype(DType::F16);
let y = model.forward(&x.to_dtype(DType::F16));
```

autocast 規則（同 PyTorch）：matmul / Linear / Conv 以半精度執行；softmax、log_softmax、LayerNorm、BatchNorm、loss 一律在 f32；不同浮點型別相加時提升為 f32，但 rank-0 純量會採用另一方的型別（`half * 2.0` 仍是 half）。

梯度累積：`.grad` 會在多次 `backward()` 間累加，把每個 micro-batch 的 loss 乘上 `1/k` 後各自 `backward()`，再 `step()` 一次即可（`tests/io.rs` 驗證與整批結果相同）。

## 功能

| 類別 | 內容 |
|---|---|
| DType | `F32`、`F16`、`BF16`（皆可微分；半精度運算內部以 f32 計算後再捨入）、`I32`（索引：token id、類別標籤、argmax 結果，存在裝置上）；可微分的 `to_dtype`、型別提升規則 |
| 推理 | **KV cache**（預先配置、容量加倍的 buffer，新 token 原地寫入、attention 直接讀 view，不再每步複製；支援 prefill + 逐步解碼、超過上下文長度時滑動視窗、clone 後寫入時複製）、`Sampler`（greedy / temperature / top-k / top-p，可設種子）、`matmul_nt`（不複製的轉置乘法，用於 tied head）|
| GPT-2 | `nn::gpt2::GPT2`：參數名稱與 Hugging Face `GPT2LMHeadModel` 相同、HF 的 Conv1D 版面 `[in, out]` 與本專案 `Linear` 一致，`model.safetensors` 不需轉換即可載入；支援 small/medium/large/xl；`tokenizer::Gpt2Tokenizer`（byte-level BPE，手寫實作 GPT-2 的前處理 regex）|
| Llama | `nn::llama::Llama`：RMSNorm、RoPE（含 Llama 3.1 頻率縮放）、SwiGLU、GQA（KV cache 只存 K/V head）、tied / untied head、int8；參數名稱同 HF `LlamaForCausalLM`，`from_pretrained`（單檔或分片 safetensors）/ `save_pretrained` |
| 視覺 | `nn::resnet::ResNet`（BasicBlock / Bottleneck、resnet18/34/50/101/152、CIFAR 式小型版；參數名稱與數量同 torchvision，`load_torchvision` 自動轉換 BN 形狀與 fc 轉置）、`nn::vit::ViT`（patch embedding、`[CLS]`、可學習位置編碼；ViT-B/16 參數數量同 timm） |
| seq2seq | `nn::seq2seq::Seq2Seq`（pre-LN encoder + 含 cross-attention 的 decoder、`encode` / `decode` / `decode_cached` / `greedy`）、`DecoderBlock`、`F::padding_mask`、`F::cross_entropy_ignore` |
| int4 | `Linear::quantize_int4(group)`、`Llama` / `GPT2::quantize_int4`：每 `group` 個輸入列共用一個 scale 的對稱 int4，一個 byte 放兩個值（約 f32 的 1/8）；`matmul_int4` 在 CPU / MPS / CUDA 有直接讀 packed 權重的融合 kernel，輸出 head 維持 int8 |
| int8 | `DType::I8`、`Linear::quantize_int8`（逐輸出通道對稱量化）、`GPT2::quantize_int8`；`matmul_int8` 原語在 CPU / MPS / CUDA 都有直接讀 int8 的融合 kernel（記憶體流量為 f32 的 1/4），大 batch 時改為反量化 + GEMM |
| 混合精度 | `autocast(F16 \| BF16, ..)`、`GradScaler`（動態 loss scaling、溢位跳步、狀態可存進 checkpoint）、`Module::to_dtype` |
| Tensor | 任意維度、NumPy 廣播、`reshape/view/permute/transpose/unsqueeze/flatten/expand`、運算子多載 (`&a + &b`, `a * 2.0`) |
| Autograd | 動態計算圖、拓撲排序反向傳播、梯度累加、`no_grad()` / `NoGradGuard` / `enable_grad()`、`detach()`；**二階導數**：`backward_create_graph()`、`autograd::grad(output, inputs, create_graph)`（不動 `.grad`，可做 Hessian-向量積、gradient penalty、MAML） |
| 運算 | `+ - * /`、`exp log sqrt powf abs tanh sigmoid relu gelu silu sin cos maximum`、`sum mean var max min argmax`、`minimum clamp where_cond masked_fill`、batched `matmul`、`index_select gather narrow select cat stack`、`im2col` |
| functional | `softmax`、`log_softmax`、`cross_entropy`（含 label smoothing）、`mse_loss`、`bce_with_logits`、`conv2d`、`max/avg_pool2d`、`max_pool2d_padded`、`layer_norm`、`rms_norm`、`dropout`、`scaled_dot_product_attention`、`causal_mask`、正弦位置編碼、RoPE（`rope_cos_sin` / `apply_rope`）、`repeat_kv` |
| nn | `Linear`、`RMSNorm`、`SwiGLU`、`Conv2d`（可無 bias）、`MaxPool2d`（可 padding）、`GlobalAvgPool`、`AvgPool2d`、`BatchNorm2d`、`LayerNorm`、`Embedding`、`Dropout`、`Flatten`、`ReLU/GELU/Tanh/Sigmoid`、`Sequential`、`FeedForward`、`MultiHeadAttention`（自注意力 / 交叉注意力）、`TransformerBlock`（Pre-LN，可選 causal）、`GPT` |
| optim | `SGD`（momentum、weight decay）、`Adam`、`AdamW`（融合 kernel，一個參數一次走訪）、`clip_grad_norm`、`cosine_lr`（含 warmup） |
| 資料 | `Dataset` trait、`DataLoader`（多執行緒預取、每 epoch 確定性打亂、`drop_last`、自訂 collate、提早中斷不卡住、worker panic 會傳遞）、`VecDataset`、`ClassificationData`、`Subset`、`random_split` |
| 存檔 | **safetensors**（Hugging Face 官方實作）：讀寫 F32/F16/BF16/I32（保留原型別），另可讀 F64/I64/I16/I8/U8/U16/U32/BOOL；`load_state_dict`（strict 或回報 missing / unexpected）；仍可讀 v0.1–v0.4 的 RSNN 格式 |
| checkpoint | 模型 + 優化器狀態（Adam 的 m/v/step、SGD 的 velocity）+ step/epoch + 全域 RNG 狀態 + 自訂 metadata，寫入時先寫暫存檔再改名；**接續訓練與不中斷訓練結果逐位元相同** |
| 排程 / 指標 | `ConstantLr`、`StepLr`、`ExponentialLr`、`CosineLr`、`LinearLr`（皆含 warmup，純函式故可從任意 step 接續）；`RunningMean`、`Ema`、`Accuracy`、`Throughput` |
| 其他 | `train()/eval()` 模式、可設定種子的亂數（`rand::Rng` 獨立串流）|

## 架構

```
src/
├── main.rs, cli/    `ruai` 指令（子指令派送、ruai nn / world / py / agent）
├── nn/              神經網路引擎（feature nn）
├── world/           Gymnasium 環境（feature world）
├── rng.rs           NumPy 相容亂數、MT19937
├── plot/            matplotlib 風格的繪圖，輸出到 HTML canvas 或 SVG（feature plot）
├── py/              py0 翻譯器（feature py，原 py2rs）
│   ├── wasm/        不經 Rust 的 py0 → WebAssembly（WasmGC）與其 JavaScript 宿主
│   └── lib/         套件 stub（*.pyi）：標準函式庫、numpy、sklearn、gymnasium、torch
├── pyrt/            py0 程式的執行期（feature pyrt）
│   └── lib/         以 Rust 實作的套件（pyrt-numpy、pyrt-sklearn、pyrt-gymnasium、pyrt-torch、pyrt-plot …）；python.rs：Python 橋接（pyrt-python）
└── agent/           以 py0 寫成的 `ruai agent`（翻譯成 cli/agent_gen.rs）
nn/
├── backend/
│   ├── mod.rs     Backend trait：約 15 個「原語」，所有裝置只需實作這些
│   ├── cpu.rs     CPU 實作（rayon 平行化）
│   ├── mps.rs     Apple Metal 後端 (--features mps)
│   ├── mps_kernels.metal  Metal compute kernels（執行時編譯）
│   ├── cuda.rs    NVIDIA CUDA 後端 (--features cuda)
│   └── cuda_kernels.cu    CUDA kernels（NVRTC 執行時編譯）
├── tensor.rs      Tensor 結構、計算圖節點、backward()
├── ops.rs         可微分運算：forward 呼叫 backend，backward 由其他運算組成
├── functional.rs  組合運算（softmax、conv2d、attention …）
├── modules/       Module trait 與各種層、Transformer、GPT、GPT-2、Llama、ResNet、ViT、Seq2Seq
├── optim/         優化器（mod.rs）與學習率排程（lr.rs）
├── data.rs        Dataset / DataLoader
├── checkpoint.rs  可接續訓練的 checkpoint
├── metrics.rs     訓練指標
└── serialize.rs   safetensors 讀寫、load_state_dict
```

**核心設計：一切都建立在少量 backend 原語上。** `Backend` trait 分兩層：

- **必要原語**（新後端一定要實作）：`from_f32 / from_i32 / to_f32 / to_i32 / full / cast`、`unary`、`binary`（含廣播）、`expand`、`where_cond`、`reduce`、`matmul`（batched + 轉置旗標）、`permute`、`narrow / cat`、`index_select / index_add`、`gather / scatter_add`、`im2col / col2im`
- **融合原語**（有預設實作，由必要原語組合而成；後端可覆寫以加速）：`softmax`、`log_softmax`、`layer_norm`（含各自的 backward）、`adam_step`

所以新後端只要完成必要原語就能完整訓練，之後再逐一覆寫融合原語來提速；`tests/backend.rs` 會自動比對融合版與預設組合版的結果。

Autograd、所有 nn 層、優化器都只透過 `Tensor` 運算操作資料，不直接碰記憶體。例如：
- `conv2d` = `im2col` + `matmul`（反向用 `col2im`）
- `max_pool2d` = `im2col` + `max`（反向用遮罩）
- 優化器在 `no_grad` 下用 Tensor 運算算出新權重，再 `set_storage` 寫回，所以在任何裝置上都能跑

**Strided view（v2.0）**：`Tensor` 是 buffer 加上 `(shape, strides, offset)`。`permute` / `transpose` / `t`、`narrow` / `select`、`expand`（broadcast 維 stride 為 0）、`unsqueeze` / `squeeze`，以及 row-major 資料的 `reshape` 都不複製資料。backend 原語仍只接受連續 buffer，`Tensor::storage()` 遇到 view 時才用 `strided_copy` 原語複製一次並快取（資料不可變，所以 view 是快照，之後 `set_storage` 不會影響它）；`matmul` 直接讀轉置與 batch-strided 的 view（例如 attention 的 `k.t()`，CPU 與 CUDA 不複製；MPS 的 GEMM 不支援自訂 batch stride，會先複製）。`is_contiguous()`、`contiguous()`、`strided_parts()` 可檢查與控制佈局。

## GPU 後端

### MPS（Apple Metal，已完成）

- **記憶體**：`MTLBuffer` 放在 unified memory（`StorageModeShared`），CPU↔GPU 傳輸只是 memcpy；kernel 的輸出緩衝區來自依大小分類的 buffer pool，避免每次配置
- **kernel**：`mps_kernels.metal` 在執行時以 safe math 編譯（不需要安裝 Xcode），數值與 CPU 版一致
- **matmul**：`MPSMatrixMultiplication`（batched + 轉置），kernel 物件依形狀快取
- **非同步**：運算累積在同一個 command buffer，每 48 個 op 或在需要讀回資料（`to_vec`、`item`）時才送出；不會每個運算都等 GPU
- **融合原語**：softmax / log_softmax / layer_norm（含 backward）、Adam 都是單一 kernel（一個 threadgroup 處理一列，用 simdgroup 歸約）
- **半精度**：kernel 以 template 對 float / half / bf16 實例化（bf16 以原始位元儲存、round-to-nearest-even），內部一律 f32 計算；matmul 的 F16 走 `MPSMatrixMultiplication` Float16，BF16 轉 f32 計算
- 需求：Apple Silicon、macOS 15+（使用 `atomic_float`、`MTLMathMode`）

### WebGPU（任何 GPU，以及瀏覽器）

- **可攜**：`--features webgpu` 建立在 wgpu 上，同一個後端在本機使用 Metal（macOS）、Vulkan（Linux、Android）或 DX12（Windows），在 wasm32 上使用瀏覽器的 WebGPU；`Device::WebGpu`、`--device webgpu`
- **Kernel**：WGSL（`webgpu_kernels.wgsl`，每個 kernel 一個模組，第一次使用時編譯）；GEMM 由程式產生並展開暫存器區塊（128×128 tile、每執行緒 8×8；小矩陣用 64×64 tile，矩陣乘向量用 split-k kernel），轉置與 batch stride 直接讀取
- **儲存**：32 位元字組；F16 / BF16 以 f32 存放，每個運算後捨入到半精度（最近偶數捨入，與 CPU 後端相同），int8 / int4 權重以位元組打包；浮點 atomic（index_add、scatter_add）用 compare-and-swap 迴圈
- **非同步**：所有 dispatch 共用一個 compute pass，每 64 個運算或讀回資料時送出；參數經由環狀緩衝區，輸出來自緩衝池（小運算約 11 µs）
- **瀏覽器**：網頁不能阻塞，所以後端以 `ruai::backend::webgpu::init().await` 建立、資料以 `tensor.to_vec_async().await` 讀回；同步讀取 GPU 資料的運算（`to_vec`、`item`）在瀏覽器中會 panic。本 crate 可用 `--features webgpu` 編譯成 `wasm32-unknown-unknown`
- **Demo**：`examples/webgpu-browser/` 在網頁中用 WebGPU 訓練字元級 GPT（Apple M3 上每步約 10 ms，與原生相同），以 `ruai::plot` 畫出 loss 並生成文字：`sh examples/webgpu-browser/build.sh && python3 -m http.server -d examples/webgpu-browser 8000`（需要 `cargo install wasm-bindgen-cli --version 0.2.130`）
- `tests/devices.rs` 逐一以 CPU 對照每個基本運算，測試套件也在其上訓練、微調與解碼（`RUSTNN_DEVICE=webgpu`）

### CUDA（NVIDIA）

- **建置**：`cudarc` 以 dynamic loading 在執行時載入 CUDA driver / NVRTC / cuBLAS，所以任何平台都能編譯 `--features cuda`；沒有 GPU 時 `Device::Cuda(0).is_available()` 回傳 `false`
- **kernel**：`cuda_kernels.cu` 是 `mps_kernels.metal` 的逐一移植（warp shuffle 歸約、`atomicAdd`），第一次使用時以 NVRTC 編譯（不開 fast-math）
- **半精度**：kernel 以 template 對 float / f16 / bf16 實例化（f16 轉換用 inline PTX `cvt`，不需要 `cuda_fp16.h`）；cuBLAS 半精度 GEMM 以 f32 累加（`CUBLAS_COMPUTE_32F`）
- **matmul**：`cublasSgemmStridedBatched`（半精度走 `GemmStridedBatchedEx`），用 `C = A·B ⇔ Cᵀ = Bᵀ·Aᵀ` 把 row-major 對應到 cuBLAS 的 column-major；batch stride 0 直接支援廣播
- **非同步**：每張卡一條 stream，記憶體為 stream-ordered（`cuMemAllocAsync`），只有讀回資料時才同步；支援多卡 `Device::Cuda(i)`
- **需求**：NVIDIA driver 與 CUDA 12.x 的 NVRTC、cuBLAS 函式庫（`libnvrtc` / `libcublas` 在載入路徑上即可，不需要完整 toolkit）

在 NVIDIA 機器上一鍵驗證（跨裝置對照、GPU 上的完整測試、範例、效能）：

```bash
./scripts/verify_cuda.sh
```

> ⚠️ CUDA 後端是在沒有 NVIDIA GPU 的環境下開發的：已驗證可編譯、kernel 語法（以 C++ shim 檢查）、kernel 名稱與參數結構對應、cuBLAS row-major 參數對應（以 Python 模擬 column-major 語意，48 種組合）、無 GPU 時的行為；**尚未在實機上執行**。

## 效能（Apple M3，8 核 CPU / 8 核 GPU，`cargo run --release --example bench <device>`）

| 項目 | v0.1 CPU | v0.3 CPU | CPU + `accelerate` | MPS (GPU) | WebGPU (GPU) |
|---|---|---|---|---|---|
| matmul 1024×1024 | 127 GFLOP/s | 283–380 GFLOP/s | 983 GFLOP/s | 1178 GFLOP/s（長時間滿載約 2000） | 760 GFLOP/s |
| softmax（1M 元素） | 1.52 ms | 0.43 ms | 0.43 ms | 0.11 ms | 0.19 ms |
| layer_norm（1M 元素） | 1.07 ms | 0.41 ms | 0.44 ms | 0.095 ms | 0.18 ms |
| conv2d fwd+bwd | 13.3 ms | 6.2 ms | 6.3 ms | 4.4 ms | 7.4 ms |
| CNN 訓練一步（小模型） | 10.1 ms | 5.9 ms | 5.1 ms | 5.5 ms | 9.4 ms |
| GPT 訓練一步（d=64, L=2） | 29.1 ms | 9.2 ms | 6.2 ms | 3.4 ms | 12.0 ms |
| GPT 訓練一步（d=256, L=4） | 296 ms | 105 ms | 58 ms | 29 ms | 69 ms |

混合精度（`bench` 的最後幾項，Apple M3）：

| 項目 | f32 | f16 | bf16 |
|---|---|---|---|
| matmul 2048×2048（MPS） | 2145 GFLOP/s | 2206 GFLOP/s | 1903 GFLOP/s（MPS 沒有 bf16 GEMM，走 f32） |
| GPT 訓練一步（d=256, L=4，MPS，autocast） | 28.7 ms | 31.9 ms | 37.0 ms |
| GPT 訓練一步（CPU，autocast） | 102 ms | 133 ms | 144 ms |

> 在 Apple M1–M3 上 FP16 與 FP32 的運算吞吐量相同，所以 autocast 在這台機器上**不會更快**（多出的型別轉換反而讓它慢約 10%）；CPU 沒有半精度運算單元，也一樣較慢。半精度的速度優勢預期出現在 NVIDIA GPU 的 Tensor Core 上（CUDA 後端已支援，尚待實機量測）；在任何裝置上它都能把權重/啟用值的記憶體減半，也能直接載入 F16/BF16 權重。

權重量化的解碼式矩陣向量乘（`[1, 2048] @ [2048, 2048]`，`bench` 最後一項，Apple M3）：

| | f32 | int8 | int4 |
|---|---|---|---|
| CPU | 0.86 ms | 0.30 ms | 0.29 ms |
| MPS | 0.22–0.33 ms | 0.08 ms | 0.17 ms |
| WebGPU | 0.26 ms | 0.16 ms | 0.15 ms |

解碼時是記憶體頻寬受限，權重越小越快（MPS 的融合 kernel 把 k 分給 8 個 simdgroup 再歸約）。

Attention 前向（`[B=4, H=8, T, D=64]`、causal，`bench` 最後一項）：

| T | CPU 融合 | CPU 組合 | MPS 融合 | MPS 組合 | 組合式的 scores 大小 |
|---|---|---|---|---|---|
| 128 | 0.84 ms | 1.26 ms | 0.86 ms | 0.60 ms | 2 MB |
| 512 | 13.4 ms | 22.4 ms | 6.3 ms | 7.3 ms | 34 MB |
| 1024 | 66.8 ms | 255.9 ms | 36.6 ms | 49.7 ms | 134 MB |

CPU 的融合 kernel 以 64×128 分塊（每塊兩個小 GEMM + online softmax），完全不存 `[Tq, Tk]` scores；MPS / CUDA 用分塊 GEMM（每塊最多 16M 個 scores），長序列記憶體有上限。

GPU 計時採 throughput 方式（連續執行 N 次後同步一次，同 `torch.cuda.synchronize()` 慣例）。很小的運算（如 256×256 matmul、小 CNN）在 GPU 上受每個 op 的派送開銷限制，不一定比 CPU 快；模型越大 GPU 優勢越明顯。

## 測試

```bash
cargo test                                         # CPU
RUSTNN_DEVICE=mps cargo test --features mps        # 同一組測試全部在 GPU 上跑
cargo test --features mps --test devices           # 每個 backend 原語：GPU 對照 CPU
```

`./test.sh` 依群組執行並在最後列出摘要：`./test.sh cpu`、`gpu`（本機所有 GPU 後端對照 CPU）、`py0`、`pyo3`、`world`、`site`、`check`（各 feature 單獨建置、wasm32 建置）、`lint`、`pretrained`、`ci`（此作業系統上 CI 跑的內容）、`all`，或用 `./test.sh file gradcheck` 只跑一個檔案；`./test.sh deps` 列出測試需要的工具。

- `tests/gradcheck.rs`：對每個可微分運算（廣播、matmul 各種 batch 形式、conv、pool、softmax、layer norm、attention …）做中央差分的數值梯度檢查，並驗證 causal mask 不會偷看未來
- `tests/backend.rs`：融合 kernel 與預設組合實作一致、matmul（含轉置、平行切塊）對照樸素實作、dtype 行為、新運算的數值
- `tests/devices.rs`：跨裝置一致性——每個原語（含邊界尺寸、廣播、各種轉置/batch stride 的 matmul、各種 stride/padding 的 conv、重複索引的 scatter）在每個可用 GPU 後端上的結果都要與 CPU 相符
- `tests/io.rs`：依 safetensors 規格**手工組出的檔案**（不經被測實作）測試多種 dtype 與 metadata、輸出檔的 header/資料區逐位元組檢查、舊格式、`load_state_dict` 報告；AdamW 與 SGD-momentum 的 checkpoint 接續後參數**逐位元相同**（含 dropout 的 RNG）；梯度累積等於整批
- `tests/data.rs`：DataLoader 順序、打亂是排列且可重現、不同 worker 數結果相同、確實平行、提早中斷、worker panic、collate；排程數值；指標
- `tests/amp.rs`：半精度數值與型別提升、可微分轉型、autocast 規則、AMP 梯度與 f32 梯度一致（F16 誤差 < 2%、BF16 < 8%）、GradScaler（還原尺度、溢位跳步、成長、狀態保存）、F32/F16/BF16 訓練都能收斂、半精度 CNN 推理、safetensors 保留半精度型別
- `tests/devices.rs` 另含所有原語的 F16/BF16 版本在 GPU 與 CPU 間的對照（含各種轉置與廣播的半精度 matmul）
- `tests/inference.rs`：KV cache 逐步／分段 prefill 的 logits 與完整前向一致、有無 cache 的生成結果相同（含滑動視窗）、GPT-2 的 HF 參數名稱與 checkpoint 載入（含 `transformer.` 前綴與 attention mask buffer）、int8 精度、BPE（自建詞表的合併與任意 Unicode 位元組級來回）、前處理 regex 的邊界情況、取樣器分佈與可重現性；**有 `data/gpt2` 時**另跑 HF 官方 GPT-2 的黃金值測試
- `tests/devices.rs` 另含 int8 matmul（融合 kernel 與 GEMM 兩條路徑）與 i8 轉型在 GPU 與 CPU 間的對照
- `tests/llama.rs`：RMSNorm 數值、RoPE 保持範數且內積只依相對位置、`repeat_kv` 版面、llama3 頻率縮放、HF `config.json` 解析與往返、GQA 下 KV cache 逐步／分段 prefill 與完整前向一致、因果性與 batch 一致、有無 cache 的生成相同、HF 版面（`[out, in]` 形狀與逐元素轉置）存檔／載入、學會序列、int8、autocast 梯度與 f32 一致
- `tests/vision.rs`：ResNet-18/34/50 與 ViT-B/16 的參數數量與 torchvision / timm 完全一致、ResNet 參數名稱與形狀、torchvision 版面（1-D BN、`[out, in]` fc、`num_batches_tracked`）載入後輸出相同、max pool 的 padding 是 -inf、小型 ResNet / ViT 學會分類
- `tests/seq2seq.rs`：解碼端 KV cache 與 teacher forcing 一致、有 padding mask 時來源 padding 完全不影響輸出（沒 mask 時會影響）、`cross_entropy_ignore`、學會反轉序列
- `tests/finetune.rs`：LoRA 初始為恆等、只訓練 adapter、凍結參數不變、合併前後輸出相同、只存 adapter 再載入、int8 底模上可訓練；gradient checkpointing 在 Llama / Llama+LoRA / 含 dropout 的 GPT / GPT-2 上梯度與不重算時相同
- `tests/quant.rs`：int4 打包與獨立計算的反量化逐元素相同、融合 kernel 與預設（主機反量化 + GEMM）一致、Llama / GPT-2 的 int4 誤差、int4 底模上的 LoRA 可訓練；`tests/devices.rs` 另含 `matmul_int4` / `dequant_int4` 在 GPU 與 CPU 的對照
- `tests/attention.rs`：融合 attention 與組合式在各種 mask（無、causal、padding、每個 head、完整、單列）、`dv ≠ d`、半精度、autocast 下一致，梯度一致，任意分塊大小結果相同，長序列大 logits 數值穩定；`tests/devices.rs` 另含 GPU 與 CPU 的 attention 對照
- `tests/views.rs` / `tests/kvcache.rs`：view 不複製且與主機計算一致、view 是快照、matmul 直接讀轉置與 strided view、各 dtype；KV cache 容量加倍、回傳 view、先前的 view 與 clone 不受之後寫入影響、梯度可經 cache 回傳、100 步解碼與完整前向一致
- `tests/second_order.rs`：逐元素、各種 matmul、softmax / log_softmax / cross-entropy、layer / RMS norm、embedding + gather、conv2d、融合 attention、gradient checkpointing 的 Hessian-向量積與有限差分一致；`backward_create_graph` 與 `autograd::grad` 的語意；gradient penalty 傳到參數的梯度與數值微分一致
- `tests/world.rs`：6 個環境 × 4 個 seed × 400 步（含 autoreset）與 gymnasium 逐步、逐位元比對，向量環境、FrozenLake 文字算繪、NumPy 亂數（含 2⁶³ 以上的 seed、各種 `integers` 範圍）
- `tests/agent.rs`：`ruai agent` 的嵌入翻譯是最新的；離線（echo 模型）約 50 個指令與互動對話、以及對假 LLM 伺服器（`tests/agent/fake_llm.py`：OpenAI / Anthropic、串流與否、原生 tool call、HTTP 500 重試）的輸出都與 CPython 執行同一份 py0 原始碼逐字相同
- `tests/py0.rs` + `tests/py0/`：py2rs 的語料——259 支程式（含 `tests/py0/numpy/`、`sklearn/`、`gymnasium/`、`torch/` 分別對照真的 NumPy、scikit-learn、Gymnasium、PyTorch；每支程式在有它所需套件的 Python 上對照）逐一翻譯、編譯、執行，輸出、exit code 與例外類型都要與 CPython 3.12+ 相同；46 支應被拒絕的程式要以預期訊息拒絕；`tests/py0/pyo3/run.sh` 檢查 PyO3 擴充模組
- `tests/training.rs`：MLP 學會 XOR、CNN（含 BatchNorm）收斂、GPT 記住序列、存檔/載入 round-trip

## 版本規劃

| 版本 | 內容 | 狀態 |
|---|---|---|
| v0.1 | CPU 後端、autograd、CNN、Transformer/GPT、優化器、存檔 | ✅ |
| v0.2 | DType（I32 索引）、cat/gather/where 等新運算、融合原語、快速 GEMM、Accelerate | ✅ |
| v0.3 | MPS（Apple Metal）後端、跨裝置測試框架、`--device` 選擇 | ✅ |
| v0.4 | CUDA 後端（cuBLAS + NVRTC）、多卡 | ✅（待實機驗證） |
| v0.5 | DataLoader、完整 checkpoint（含優化器狀態與 RNG）、safetensors、LR 排程、梯度累積、指標 | ✅ |
| v0.6 | f16 / bf16（CPU / MPS / CUDA）、autocast、GradScaler、半精度推理 | ✅ |
| v0.7 | KV cache、載入 GPT-2 官方權重、BPE tokenizer、取樣、int8 量化 | ✅（v2.0.1 以官方權重驗證） |
| v0.8 | RMSNorm、RoPE、SwiGLU、GQA、Llama 模型（HF 版面載入／匯出） | ✅（v2.0.1 以 SmolLM2 官方權重驗證） |
| v0.9 | ResNet、ViT、encoder-decoder Transformer（cross-attention、seq2seq） | ✅ |
| v1.0 | API 整理（`GPT::generate` 改用 `Sampler`）、所有公開項目都有 rustdoc、clippy 乾淨、`cargo package` 可打包 | ✅（crates.io 名稱：`ruai`） |
| v1.1 | GitHub Actions CI（Linux / macOS，可手動觸發 GPT-2 真實權重測試）、`zeros_like` 保留 dtype、`collator` 保留影像形狀 | ✅ |
| v1.2 | Hugging Face `tokenizer.json` 分詞器（`HfTokenizer`）、`llama --model` 載入預訓練模型 | ✅（v2.0.1 以 GPT-2 / SmolLM2 的檔案驗證） |
| v1.3 | LoRA（含 int8 底模）、gradient checkpointing、`freeze` / `trainable_parameters` | ✅ |
| v1.4 | 分組 int4 權重量化（CPU / MPS / CUDA 融合 kernel，可在其上訓練 LoRA） | ✅ |
| v1.5 | 融合 attention（CPU：FlashAttention 式分塊；其他後端：分塊 GEMM，記憶體有上限）、MPS 量化 matvec 加速 | ✅ |
| v2.0 | 不複製的 strided view、預先配置的 KV cache（解碼 CPU 約 1.4 倍、MPS 約 6 倍）、二階導數（`create_graph`、`autograd::grad`） | ✅ |
| v3.0 | 單一 crate + features、引擎移到 `src/nn/`、git 式 `ruai` 指令（`ruai nn devices / info / generate`、外部 `ruai-NAME`） | ✅ |
| v3.1 | `world`：gymnasium 1.x 的 API 與環境（逐位元相同）、NumPy 相容亂數 `ruai::rng`、`ruai world list / run`、REINFORCE 範例 | ✅ |
| v3.2 | `py`：以 git subtree 併入 py2rs（翻譯器 `ruai::py`、執行期 `ruai::pyrt`），`ruai py run / emit / check / explain / build / pyo3 / stub / test / fuzz` | ✅ |
| v3.3 | py0 套件機制（`import M [as A]` / `from M import x` → `src/py/lib/M.pyi` + `ruai::pyrt::lib::M`）、以 Rust 重寫的 Python 標準函式庫（第一批：`shlex`、`fnmatch`、`html`） | ✅ |
| v3.4 | `ruai agent`：以 py0 改寫 agge 並嵌入 `ruai`；py0 新增 `os` / `sys` / `time` 擴充、`subprocess`、`ruai.json`、`ruai.http`、`Result[T, OSError]` | ✅ |
| v3.5 | py0 的 `numpy` 子集（ndarray、廣播、索引與 view、歸約、`linalg`、`random`，輸出與 NumPy 逐字相同）；stub 的 union 參數與 overload | ✅ |
| v3.6 | py0 的 `sklearn` 子集（線性模型、羅吉斯迴歸、k-NN、naive Bayes、KMeans、PCA、前處理、`train_test_split`、指標、iris） | ✅ |
| v3.7 | py0 的 `gymnasium`（world 環境，與 Gymnasium 逐位元相同）與 `torch` 子集（nn 引擎上的 autograd、`nn` 模組、損失、`optim`，`manual_seed` 與 PyTorch 相同的亂數）；numpy 支援 float32 | ✅ |
| v4.0 | 穩定版：語意化版本（見「穩定性」）、英文文件 | ✅ |
| v4.1 | Python 橋接：py0 透過 stub（`# python: MODULE`）呼叫 CPython 函式庫，`ruai py stub --python` | ✅ |
| v4.2 | `ruai.plot`：matplotlib pyplot 的 API，畫在瀏覽器 canvas 或 SVG | ✅ |
| v4.3 | 不經 Rust 的 py0 → WebAssembly（WasmGC），py0 編譯器在瀏覽器中執行 | ✅ |
| v4.4 | nn 的 WebGPU 後端（wgpu：Metal / Vulkan / DX12 與瀏覽器），在網頁中訓練 GPT | ✅ |
| v4.5 | GitHub Pages 網站：VS Code 風格的 py0 IDE、可匯出 `.ipynb` 的 py0 notebook、各種展示；`run.sh`、`test.sh` | ✅ |

## 穩定性

從 4.0 起，以下部分遵循語意化版本（semver）：

- `ruai::nn`（也 re-export 在 crate 根目錄）、`ruai::world`、`ruai::rng` 的 Rust API；
- `ruai` 指令（子指令與選項）；
- py0 語言（[py0/SPEC.md](py0/SPEC.md)）與其套件的行為：在某個 4.x 能執行的程式，在之後的 4.x 也能執行且輸出相同。

`ruai::py`（翻譯器）與 `ruai::pyrt`（翻譯出的程式呼叫的執行期）因為產生的程式碼要用而是公開的，但它們的 Rust API 跟著翻譯器走，次版本間可能改變；翻譯出的程式依賴對應版本的 ruai。

## 目前限制

- 半精度在 CPU 上是儲存格式（內部轉 f32 計算），不會更快
- `index_add` / `scatter_add` / 融合 kernel 在 GPU 上只有 f32 版本，半精度輸入會先轉 f32（autocast 下本來就以 f32 執行）
- 二階導數經過融合 attention 與 gradient checkpointing 時會重算一次前向
- GPU 上的 attention 是分塊 GEMM，不是單一融合 kernel（真正的 GPU FlashAttention 需要 simdgroup 矩陣分塊）；backward 會重算並暫時建立 scores
- MPS：小運算受派送開銷限制
- CUDA 後端尚未在 NVIDIA 實機上執行過（見上方說明與 `scripts/verify_cuda.sh`）
- `HfTokenizer` 只支援 BPE 模型；Llama 3 的 `rope_scaling` 與 Llama 2 式（SentencePiece）`tokenizer.json` 尚未以真實檔案驗證
- MPS / CUDA：張量維度上限 8；元素數必須能放進 u32
