# rust-nn

一個用 Rust 寫的、類似 PyTorch 的神經網路引擎：動態計算圖 + 反向自動微分，支援訓練與推理，內建 CNN 與 Transformer（GPT）。支援 CPU（rayon 多執行緒，macOS 可選用 Accelerate）、Apple GPU（Metal / MPS）與 NVIDIA GPU（CUDA）。

## 快速開始

```bash
cargo test                              # 梯度數值檢查 + 端到端訓練測試
cargo run --release --example cnn       # CNN 影像分類（有 MNIST 就用 MNIST）
cargo run --release --example gpt       # 字元級 GPT：訓練 → 存檔 → 載入 → 生成
cargo run --release --example bench     # 效能基準測試

# macOS：用 Apple Accelerate (AMX) 做矩陣乘法，大型模型約快 2 倍
cargo run --release --features accelerate --example gpt

# Apple GPU（Metal）：加上 --features mps，用 --device 選裝置（cpu / mps / auto）
cargo run --release --features mps --example gpt -- --device mps
cargo run --release --features mps --example bench mps

# NVIDIA GPU（CUDA）：加上 --features cuda
cargo run --release --features cuda --example gpt -- --device cuda
```

程式裡選裝置：`Tensor::randn(&[2, 3], Device::Mps)`、`model.to_device(Device::Mps)`、`Device::best_available()`，或用環境變數 `RUSTNN_DEVICE=mps`。

MNIST：把解壓後的 `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte` 放到 `data/mnist/`；找不到時 `cnn` 範例會改用自動產生的幾何圖形資料集。

## 使用範例

```rust
use rustnn::prelude::*;
use rustnn::nn::{Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear};
use rustnn::optim::Adam;

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

// 存檔 / 載入
rustnn::serialize::save("model.rsnn", &model)?;
rustnn::serialize::load("model.rsnn", &model)?;
```

## 功能

| 類別 | 內容 |
|---|---|
| DType | `F32`（所有可微分運算）、`I32`（索引：token id、類別標籤、argmax 結果），索引 tensor 存在裝置上 |
| Tensor | 任意維度、NumPy 廣播、`reshape/view/permute/transpose/unsqueeze/flatten/expand`、運算子多載 (`&a + &b`, `a * 2.0`) |
| Autograd | 動態計算圖、拓撲排序反向傳播、梯度累加、`no_grad()` / `NoGradGuard`、`detach()` |
| 運算 | `+ - * /`、`exp log sqrt powf abs tanh sigmoid relu gelu sin cos maximum`、`sum mean var max min argmax`、`minimum clamp where_cond masked_fill`、batched `matmul`、`index_select gather narrow select cat stack`、`im2col` |
| functional | `softmax`、`log_softmax`、`cross_entropy`（含 label smoothing）、`mse_loss`、`bce_with_logits`、`conv2d`、`max/avg_pool2d`、`layer_norm`、`dropout`、`scaled_dot_product_attention`、`causal_mask`、正弦位置編碼 |
| nn | `Linear`、`Conv2d`、`MaxPool2d`、`AvgPool2d`、`BatchNorm2d`、`LayerNorm`、`Embedding`、`Dropout`、`Flatten`、`ReLU/GELU/Tanh/Sigmoid`、`Sequential`、`FeedForward`、`MultiHeadAttention`（自注意力 / 交叉注意力）、`TransformerBlock`（Pre-LN，可選 causal）、`GPT` |
| optim | `SGD`（momentum、weight decay）、`Adam`、`AdamW`（融合 kernel，一個參數一次走訪）、`clip_grad_norm`、`cosine_lr`（含 warmup） |
| 其他 | 模型存檔/載入（含 BatchNorm running stats）、`train()/eval()` 模式、可設定種子的亂數 |

## 架構

```
src/
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
├── nn/            Module trait 與各種層、Transformer、GPT
├── optim.rs       優化器
└── serialize.rs   權重存檔格式
```

**核心設計：一切都建立在少量 backend 原語上。** `Backend` trait 分兩層：

- **必要原語**（新後端一定要實作）：`from_f32 / from_i32 / to_f32 / to_i32 / full / cast`、`unary`、`binary`（含廣播）、`expand`、`where_cond`、`reduce`、`matmul`（batched + 轉置旗標）、`permute`、`narrow / cat`、`index_select / index_add`、`gather / scatter_add`、`im2col / col2im`
- **融合原語**（有預設實作，由必要原語組合而成；後端可覆寫以加速）：`softmax`、`log_softmax`、`layer_norm`（含各自的 backward）、`adam_step`

所以新後端只要完成必要原語就能完整訓練，之後再逐一覆寫融合原語來提速；`tests/backend.rs` 會自動比對融合版與預設組合版的結果。

Autograd、所有 nn 層、優化器都只透過 `Tensor` 運算操作資料，不直接碰記憶體。例如：
- `conv2d` = `im2col` + `matmul`（反向用 `col2im`）
- `max_pool2d` = `im2col` + `max`（反向用遮罩）
- 優化器在 `no_grad` 下用 Tensor 運算算出新權重，再 `set_storage` 寫回，所以在任何裝置上都能跑

所有 storage 都是連續 (contiguous) 的 row-major buffer，這讓 GPU kernel 的實作保持簡單。

## GPU 後端

### MPS（Apple Metal，已完成）

- **記憶體**：`MTLBuffer` 放在 unified memory（`StorageModeShared`），CPU↔GPU 傳輸只是 memcpy；kernel 的輸出緩衝區來自依大小分類的 buffer pool，避免每次配置
- **kernel**：`mps_kernels.metal` 在執行時以 safe math 編譯（不需要安裝 Xcode），數值與 CPU 版一致
- **matmul**：`MPSMatrixMultiplication`（batched + 轉置），kernel 物件依形狀快取
- **非同步**：運算累積在同一個 command buffer，每 48 個 op 或在需要讀回資料（`to_vec`、`item`）時才送出；不會每個運算都等 GPU
- **融合原語**：softmax / log_softmax / layer_norm（含 backward）、Adam 都是單一 kernel（一個 threadgroup 處理一列，用 simdgroup 歸約）
- 需求：Apple Silicon、macOS 15+（使用 `atomic_float`、`MTLMathMode`）

### CUDA（NVIDIA）

- **建置**：`cudarc` 以 dynamic loading 在執行時載入 CUDA driver / NVRTC / cuBLAS，所以任何平台都能編譯 `--features cuda`；沒有 GPU 時 `Device::Cuda(0).is_available()` 回傳 `false`
- **kernel**：`cuda_kernels.cu` 是 `mps_kernels.metal` 的逐一移植（warp shuffle 歸約、`atomicAdd`），第一次使用時以 NVRTC 編譯（不開 fast-math）
- **matmul**：`cublasSgemmStridedBatched`，用 `C = A·B ⇔ Cᵀ = Bᵀ·Aᵀ` 把 row-major 對應到 cuBLAS 的 column-major；batch stride 0 直接支援廣播
- **非同步**：每張卡一條 stream，記憶體為 stream-ordered（`cuMemAllocAsync`），只有讀回資料時才同步；支援多卡 `Device::Cuda(i)`
- **需求**：NVIDIA driver 與 CUDA 12.x 的 NVRTC、cuBLAS 函式庫（`libnvrtc` / `libcublas` 在載入路徑上即可，不需要完整 toolkit）

在 NVIDIA 機器上一鍵驗證（跨裝置對照、GPU 上的完整測試、範例、效能）：

```bash
./scripts/verify_cuda.sh
```

> ⚠️ CUDA 後端是在沒有 NVIDIA GPU 的環境下開發的：已驗證可編譯、kernel 語法（以 C++ shim 檢查）、kernel 名稱與參數結構對應、cuBLAS row-major 參數對應（以 Python 模擬 column-major 語意，48 種組合）、無 GPU 時的行為；**尚未在實機上執行**。

## 效能（Apple M3，8 核 CPU / 8 核 GPU，`cargo run --release --example bench <device>`）

| 項目 | v0.1 CPU | v0.3 CPU | CPU + `accelerate` | MPS (GPU) |
|---|---|---|---|---|
| matmul 1024×1024 | 127 GFLOP/s | 283–380 GFLOP/s | 983 GFLOP/s | 1178 GFLOP/s（長時間滿載約 2000） |
| softmax（1M 元素） | 1.52 ms | 0.43 ms | 0.43 ms | 0.11 ms |
| layer_norm（1M 元素） | 1.07 ms | 0.41 ms | 0.44 ms | 0.095 ms |
| conv2d fwd+bwd | 13.3 ms | 6.2 ms | 6.3 ms | 4.4 ms |
| CNN 訓練一步（小模型） | 10.1 ms | 5.9 ms | 5.1 ms | 5.5 ms |
| GPT 訓練一步（d=64, L=2） | 29.1 ms | 9.2 ms | 6.2 ms | 3.4 ms |
| GPT 訓練一步（d=256, L=4） | 296 ms | 105 ms | 58 ms | 29 ms |

GPU 計時採 throughput 方式（連續執行 N 次後同步一次，同 `torch.cuda.synchronize()` 慣例）。很小的運算（如 256×256 matmul、小 CNN）在 GPU 上受每個 op 的派送開銷限制，不一定比 CPU 快；模型越大 GPU 優勢越明顯。

## 測試

```bash
cargo test                                         # CPU
RUSTNN_DEVICE=mps cargo test --features mps        # 同一組測試全部在 GPU 上跑
cargo test --features mps --test devices           # 每個 backend 原語：GPU 對照 CPU
```

- `tests/gradcheck.rs`：對每個可微分運算（廣播、matmul 各種 batch 形式、conv、pool、softmax、layer norm、attention …）做中央差分的數值梯度檢查，並驗證 causal mask 不會偷看未來
- `tests/backend.rs`：融合 kernel 與預設組合實作一致、matmul（含轉置、平行切塊）對照樸素實作、dtype 行為、新運算的數值
- `tests/devices.rs`：跨裝置一致性——每個原語（含邊界尺寸、廣播、各種轉置/batch stride 的 matmul、各種 stride/padding 的 conv、重複索引的 scatter）在每個可用 GPU 後端上的結果都要與 CPU 相符
- `tests/training.rs`：MLP 學會 XOR、CNN（含 BatchNorm）收斂、GPT 記住序列、存檔/載入 round-trip

## 版本規劃

| 版本 | 內容 | 狀態 |
|---|---|---|
| v0.1 | CPU 後端、autograd、CNN、Transformer/GPT、優化器、存檔 | ✅ |
| v0.2 | DType（I32 索引）、cat/gather/where 等新運算、融合原語、快速 GEMM、Accelerate | ✅ |
| v0.3 | MPS（Apple Metal）後端、跨裝置測試框架、`--device` 選擇 | ✅ |
| v0.4 | CUDA 後端（cuBLAS + NVRTC）、多卡 | ✅（待實機驗證） |
| v0.5 | DataLoader、完整 checkpoint（含優化器狀態）、safetensors | |
| v0.6 | f16 / bf16、混合精度訓練 | |
| v0.7 | KV cache、載入 GPT-2 官方權重、int8 量化 | |
| v0.8 | ResNet、ViT、encoder-decoder、RMSNorm/RoPE/SwiGLU | |
| v1.0 | API 穩定、文件、發布 crates.io | |

## 目前限制

- 浮點運算只有 `f32`（f16/bf16 在 v0.6）
- 反向傳播不建立計算圖（不支援二階導數）
- attention 尚未融合成單一 kernel（flash-attention 式的融合排在之後的版本）
- MPS：小運算受派送開銷限制
- CUDA 後端尚未在 NVIDIA 實機上執行過（見上方說明與 `scripts/verify_cuda.sh`）
- MPS / CUDA：張量維度上限 8；元素數必須能放進 u32
