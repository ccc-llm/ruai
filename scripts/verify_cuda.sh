#!/usr/bin/env bash
# Verify the CUDA backend on a machine with an NVIDIA GPU.
# Needs: NVIDIA driver, and the CUDA 12.x runtime libraries NVRTC + cuBLAS
# (libnvrtc.so / libcublas.so on the loader path; the full toolkit is not required).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. GPU vs CPU, primitive by primitive"
cargo test --release --features cuda --test devices -- --nocapture

echo "== 2. full gradcheck + training suites on the GPU"
RUSTNN_DEVICE=cuda cargo test --release --features cuda

echo "== 3. examples on the GPU"
cargo run --release --features cuda --example cnn -- --device cuda
cargo run --release --features cuda --example gpt -- --device cuda

echo "== 4. benchmarks"
cargo run --release --example bench cpu
cargo run --release --features cuda --example bench cuda
