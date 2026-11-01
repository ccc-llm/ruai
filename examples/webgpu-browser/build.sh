#!/bin/sh
# Build the demo: the wasm module and its JS bindings in pkg/ (needs
# `cargo install wasm-bindgen-cli --version 0.2.130`). Then serve the directory:
#   sh examples/webgpu-browser/build.sh && python3 -m http.server -d examples/webgpu-browser 8000
set -e
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir pkg target/wasm32-unknown-unknown/release/ruai_webgpu_demo.wasm
ls -l pkg
