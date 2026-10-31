#!/bin/sh
# Build the py0 playground: the compiler as WebAssembly (ruai_py0.wasm) and the
# host of compiled programs (ruai_wasm.mjs). Serve the directory over HTTP:
#   sh py0/web/build.sh && python3 -m http.server -d py0/web 8000
set -e
cd "$(dirname "$0")/../.."
cargo build --release --example py0_wasm --target wasm32-unknown-unknown --no-default-features --features py
cp target/wasm32-unknown-unknown/release/examples/py0_wasm.wasm py0/web/ruai_py0.wasm
cp src/py/wasm/host.mjs py0/web/ruai_wasm.mjs
ls -l py0/web/ruai_py0.wasm
