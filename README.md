# ruai（rust-nn）

[![CI](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml/badge.svg)](https://github.com/ccc-llm/ruai/actions/workflows/ci.yml)

一個用 Rust 寫的、類似 PyTorch 的神經網路引擎：動態計算圖 + 反向自動微分，支援訓練與推理，內建 CNN、ResNet、ViT、Transformer（GPT / GPT-2 / Llama）與 encoder-decoder。支援 CPU（rayon 多執行緒，macOS 可選用 Accelerate）、Apple GPU（Metal / MPS）與 NVIDIA GPU（CUDA）。

## 安裝

```toml
[dependencies]
ruai = "1.0"   # 或 git 版本：ruai = { git = "https://github.com/ccc-llm/ruai" }
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

> ⚠️ 尚未以真實 GPT-2 權重驗證：載入、KV cache、tokenizer 與 int8 都已用合成資料測試（見下方「測試」），但還沒有對官方權重執行過。下載後執行 `cargo test --release --test inference` 會自動跑兩個黃金值測試：tokenizer（`"The dog"` → `[464, 3290]`）與 greedy 生成要與 Hugging Face transformers 的 GPT-2 整合測試（`test_lm_generate_gpt2`）結果一致。該期望值是依記憶寫入的，若不符請先核對期望值本身。

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

> ⚠️ 尚未以真實 Llama 權重驗證：HF 版面（名稱、形狀、轉置）、RoPE 性質、llama3 頻率縮放與 config 解析都以合成資料測試，但還沒有對官方權重比對過 logits。專案目前沒有 SentencePiece / `tokenizer.json` 分詞器，載入預訓練 Llama 後需自行提供 token id。

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
| 推理 | **KV cache**（逐 token 解碼為 O(n)，支援 prefill + 逐步解碼、超過上下文長度時滑動視窗）、`Sampler`（greedy / temperature / top-k / top-p，可設種子）、`matmul_nt`（不複製的轉置乘法，用於 tied head）|
| GPT-2 | `nn::gpt2::GPT2`：參數名稱與 Hugging Face `GPT2LMHeadModel` 相同、HF 的 Conv1D 版面 `[in, out]` 與本專案 `Linear` 一致，`model.safetensors` 不需轉換即可載入；支援 small/medium/large/xl；`tokenizer::Gpt2Tokenizer`（byte-level BPE，手寫實作 GPT-2 的前處理 regex）|
| Llama | `nn::llama::Llama`：RMSNorm、RoPE（含 Llama 3.1 頻率縮放）、SwiGLU、GQA（KV cache 只存 K/V head）、tied / untied head、int8；參數名稱同 HF `LlamaForCausalLM`，`from_pretrained`（單檔或分片 safetensors）/ `save_pretrained` |
| 視覺 | `nn::resnet::ResNet`（BasicBlock / Bottleneck、resnet18/34/50/101/152、CIFAR 式小型版；參數名稱與數量同 torchvision，`load_torchvision` 自動轉換 BN 形狀與 fc 轉置）、`nn::vit::ViT`（patch embedding、`[CLS]`、可學習位置編碼；ViT-B/16 參數數量同 timm） |
| seq2seq | `nn::seq2seq::Seq2Seq`（pre-LN encoder + 含 cross-attention 的 decoder、`encode` / `decode` / `decode_cached` / `greedy`）、`DecoderBlock`、`F::padding_mask`、`F::cross_entropy_ignore` |
| int8 | `DType::I8`、`Linear::quantize_int8`（逐輸出通道對稱量化）、`GPT2::quantize_int8`；`matmul_int8` 原語在 CPU / MPS / CUDA 都有直接讀 int8 的融合 kernel（記憶體流量為 f32 的 1/4），大 batch 時改為反量化 + GEMM |
| 混合精度 | `autocast(F16 \| BF16, ..)`、`GradScaler`（動態 loss scaling、溢位跳步、狀態可存進 checkpoint）、`Module::to_dtype` |
| Tensor | 任意維度、NumPy 廣播、`reshape/view/permute/transpose/unsqueeze/flatten/expand`、運算子多載 (`&a + &b`, `a * 2.0`) |
| Autograd | 動態計算圖、拓撲排序反向傳播、梯度累加、`no_grad()` / `NoGradGuard`、`detach()` |
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
├── nn/            Module trait 與各種層、Transformer、GPT、GPT-2、Llama、ResNet、ViT、Seq2Seq
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

所有 storage 都是連續 (contiguous) 的 row-major buffer，這讓 GPU kernel 的實作保持簡單。

## GPU 後端

### MPS（Apple Metal，已完成）

- **記憶體**：`MTLBuffer` 放在 unified memory（`StorageModeShared`），CPU↔GPU 傳輸只是 memcpy；kernel 的輸出緩衝區來自依大小分類的 buffer pool，避免每次配置
- **kernel**：`mps_kernels.metal` 在執行時以 safe math 編譯（不需要安裝 Xcode），數值與 CPU 版一致
- **matmul**：`MPSMatrixMultiplication`（batched + 轉置），kernel 物件依形狀快取
- **非同步**：運算累積在同一個 command buffer，每 48 個 op 或在需要讀回資料（`to_vec`、`item`）時才送出；不會每個運算都等 GPU
- **融合原語**：softmax / log_softmax / layer_norm（含 backward）、Adam 都是單一 kernel（一個 threadgroup 處理一列，用 simdgroup 歸約）
- **半精度**：kernel 以 template 對 float / half / bf16 實例化（bf16 以原始位元儲存、round-to-nearest-even），內部一律 f32 計算；matmul 的 F16 走 `MPSMatrixMultiplication` Float16，BF16 轉 f32 計算
- 需求：Apple Silicon、macOS 15+（使用 `atomic_float`、`MTLMathMode`）

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

| 項目 | v0.1 CPU | v0.3 CPU | CPU + `accelerate` | MPS (GPU) |
|---|---|---|---|---|
| matmul 1024×1024 | 127 GFLOP/s | 283–380 GFLOP/s | 983 GFLOP/s | 1178 GFLOP/s（長時間滿載約 2000） |
| softmax（1M 元素） | 1.52 ms | 0.43 ms | 0.43 ms | 0.11 ms |
| layer_norm（1M 元素） | 1.07 ms | 0.41 ms | 0.44 ms | 0.095 ms |
| conv2d fwd+bwd | 13.3 ms | 6.2 ms | 6.3 ms | 4.4 ms |
| CNN 訓練一步（小模型） | 10.1 ms | 5.9 ms | 5.1 ms | 5.5 ms |
| GPT 訓練一步（d=64, L=2） | 29.1 ms | 9.2 ms | 6.2 ms | 3.4 ms |
| GPT 訓練一步（d=256, L=4） | 296 ms | 105 ms | 58 ms | 29 ms |

混合精度（`bench` 的最後幾項，Apple M3）：

| 項目 | f32 | f16 | bf16 |
|---|---|---|---|
| matmul 2048×2048（MPS） | 2145 GFLOP/s | 2206 GFLOP/s | 1903 GFLOP/s（MPS 沒有 bf16 GEMM，走 f32） |
| GPT 訓練一步（d=256, L=4，MPS，autocast） | 28.7 ms | 31.9 ms | 37.0 ms |
| GPT 訓練一步（CPU，autocast） | 102 ms | 133 ms | 144 ms |

> 在 Apple M1–M3 上 FP16 與 FP32 的運算吞吐量相同，所以 autocast 在這台機器上**不會更快**（多出的型別轉換反而讓它慢約 10%）；CPU 沒有半精度運算單元，也一樣較慢。半精度的速度優勢預期出現在 NVIDIA GPU 的 Tensor Core 上（CUDA 後端已支援，尚待實機量測）；在任何裝置上它都能把權重/啟用值的記憶體減半，也能直接載入 F16/BF16 權重。

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
- `tests/io.rs`：依 safetensors 規格**手工組出的檔案**（不經被測實作）測試多種 dtype 與 metadata、輸出檔的 header/資料區逐位元組檢查、舊格式、`load_state_dict` 報告；AdamW 與 SGD-momentum 的 checkpoint 接續後參數**逐位元相同**（含 dropout 的 RNG）；梯度累積等於整批
- `tests/data.rs`：DataLoader 順序、打亂是排列且可重現、不同 worker 數結果相同、確實平行、提早中斷、worker panic、collate；排程數值；指標
- `tests/amp.rs`：半精度數值與型別提升、可微分轉型、autocast 規則、AMP 梯度與 f32 梯度一致（F16 誤差 < 2%、BF16 < 8%）、GradScaler（還原尺度、溢位跳步、成長、狀態保存）、F32/F16/BF16 訓練都能收斂、半精度 CNN 推理、safetensors 保留半精度型別
- `tests/devices.rs` 另含所有原語的 F16/BF16 版本在 GPU 與 CPU 間的對照（含各種轉置與廣播的半精度 matmul）
- `tests/inference.rs`：KV cache 逐步／分段 prefill 的 logits 與完整前向一致、有無 cache 的生成結果相同（含滑動視窗）、GPT-2 的 HF 參數名稱與 checkpoint 載入（含 `transformer.` 前綴與 attention mask buffer）、int8 精度、BPE（自建詞表的合併與任意 Unicode 位元組級來回）、前處理 regex 的邊界情況、取樣器分佈與可重現性；**有 `data/gpt2` 時**另跑 HF 官方 GPT-2 的黃金值測試
- `tests/devices.rs` 另含 int8 matmul（融合 kernel 與 GEMM 兩條路徑）與 i8 轉型在 GPU 與 CPU 間的對照
- `tests/llama.rs`：RMSNorm 數值、RoPE 保持範數且內積只依相對位置、`repeat_kv` 版面、llama3 頻率縮放、HF `config.json` 解析與往返、GQA 下 KV cache 逐步／分段 prefill 與完整前向一致、因果性與 batch 一致、有無 cache 的生成相同、HF 版面（`[out, in]` 形狀與逐元素轉置）存檔／載入、學會序列、int8、autocast 梯度與 f32 一致
- `tests/vision.rs`：ResNet-18/34/50 與 ViT-B/16 的參數數量與 torchvision / timm 完全一致、ResNet 參數名稱與形狀、torchvision 版面（1-D BN、`[out, in]` fc、`num_batches_tracked`）載入後輸出相同、max pool 的 padding 是 -inf、小型 ResNet / ViT 學會分類
- `tests/seq2seq.rs`：解碼端 KV cache 與 teacher forcing 一致、有 padding mask 時來源 padding 完全不影響輸出（沒 mask 時會影響）、`cross_entropy_ignore`、學會反轉序列
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
| v0.7 | KV cache、載入 GPT-2 官方權重、BPE tokenizer、取樣、int8 量化 | ✅（GPT-2 尚未以真實權重驗證） |
| v0.8 | RMSNorm、RoPE、SwiGLU、GQA、Llama 模型（HF 版面載入／匯出） | ✅（Llama 尚未以真實權重驗證） |
| v0.9 | ResNet、ViT、encoder-decoder Transformer（cross-attention、seq2seq） | ✅ |
| v1.0 | API 整理（`GPT::generate` 改用 `Sampler`）、所有公開項目都有 rustdoc、clippy 乾淨、`cargo package` 可打包 | ✅（crates.io 名稱：`ruai`） |
| v1.1 | GitHub Actions CI（Linux / macOS，可手動觸發 GPT-2 真實權重測試）、`zeros_like` 保留 dtype、`collator` 保留影像形狀 | ✅ |
| v1.2 | Hugging Face `tokenizer.json` 分詞器 | |
| v1.3 | 預先配置的 KV cache（O(n) 解碼）、LoRA、gradient checkpointing | |
| v1.4 | int4 權重量化 | |
| v1.5 | 融合 attention kernel | |
| v2.0 | 不複製的 strided view、二階導數 | |

## 目前限制

- 半精度在 CPU 上是儲存格式（內部轉 f32 計算），不會更快
- `index_add` / `scatter_add` / 融合 kernel 在 GPU 上只有 f32 版本，半精度輸入會先轉 f32（autocast 下本來就以 f32 執行）
- 反向傳播不建立計算圖（不支援二階導數）
- attention 尚未融合成單一 kernel（flash-attention 式的融合排在之後的版本）
- MPS：小運算受派送開銷限制
- CUDA 後端尚未在 NVIDIA 實機上執行過（見上方說明與 `scripts/verify_cuda.sh`）
- GPT-2 尚未以真實權重驗證（見上方說明與 `scripts/download_gpt2.sh`）
- Llama 尚未以真實權重驗證，且沒有 SentencePiece / `tokenizer.json` 分詞器
- MPS / CUDA：張量維度上限 8；元素數必須能放進 u32
