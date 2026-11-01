#!/usr/bin/env bash
# Run ruai's tests: one group, or all of them, with a summary at the end.
#   ./test.sh <group> [more groups...]
#
#   cpu            cargo test (every test file, on the CPU)
#   gpu            the GPU backends against the CPU (tests/devices.rs) on this machine's GPUs
#   mps            all tests with Metal built in (macOS), and Metal against the CPU
#   mps-all        all tests again, with every tensor on the Apple GPU (RUSTNN_DEVICE=mps)
#   webgpu         WebGPU against the CPU (tests/devices.rs)
#   cuda           CUDA against the CPU (needs an NVIDIA GPU)
#   py0            the py0 corpus against CPython (Rust and WebAssembly backends)
#   wasm           only the py0 → WebAssembly corpus
#   pyo3           py0 → a Python extension module, against CPython
#   world          the Gymnasium environments against Gymnasium, bit for bit
#   agent          ruai agent against CPython running the same py0 sources
#   site           build the website and run its py0 examples
#   check          build with each feature alone, for wasm32, the browser demo and the site crate
#   lint           clippy (warnings are errors) and rustdoc
#   pretrained     against real GPT-2 / SmolLM2 weights (scripts/download_*.sh first)
#   file NAME [FILTER]   one test file (tests/NAME.rs), optionally only matching tests
#   ci             what CI runs on this OS
#   all            every group this machine can run
#   deps           show which tools and Python packages the tests need
#
# A group's steps all run even when one fails; the exit code is non-zero if any failed.
set -u
cd "$(dirname "$0")"

OS="$(uname -s)"
PYTHON="${PY2RS_PYTHON:-python3}"
RESULTS=()
FAILED=0

# run one step: a name, then the command
step() {
  local name="$1"
  shift
  printf '\n\033[1m==> %s\033[0m\n$ %s\n' "$name" "$*"
  local t0=$SECONDS
  if "$@"; then
    RESULTS+=("ok    $(printf '%4ss' $((SECONDS - t0)))  $name")
  else
    RESULTS+=("FAIL  $(printf '%4ss' $((SECONDS - t0)))  $name")
    FAILED=$((FAILED + 1))
  fi
}

skip() {
  RESULTS+=("skip          $1")
}

has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

usage() {
  sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
}

group() {
  case "$1" in
    cpu) step "cargo test (CPU)" cargo test --release ;;
    gpu)
      if [ "$OS" = Darwin ]; then group mps-devices; fi
      if has_nvidia; then group cuda; fi
      group webgpu
      ;;
    mps-devices) step "Metal against the CPU" cargo test --release --features mps --test devices ;;
    mps)
      if [ "$OS" != Darwin ]; then skip "mps (macOS only)"; return; fi
      step "cargo test with mps + accelerate" cargo test --release --features mps,accelerate
      group mps-devices
      ;;
    mps-all)
      if [ "$OS" != Darwin ]; then skip "mps-all (macOS only)"; return; fi
      step "all tests on the Apple GPU" env RUSTNN_DEVICE=mps cargo test --release --features mps
      ;;
    webgpu) step "WebGPU against the CPU" cargo test --release --features webgpu --test devices ;;
    cuda)
      if ! has_nvidia; then skip "cuda (no NVIDIA GPU)"; return; fi
      step "CUDA against the CPU" cargo test --release --features cuda --test devices
      ;;
    py0) step "py0 corpus (Rust and wasm) against CPython" cargo test --release --test py0 ;;
    wasm) step "py0 → wasm corpus against CPython" cargo test --release --test py0 wasm_corpus_matches_cpython ;;
    pyo3)
      step "build ruai" cargo build --release
      step "py0 → PyO3 module against CPython" env PY2RS_PYTHON="$PYTHON" sh tests/py0/pyo3/run.sh
      ;;
    world) step "environments against Gymnasium" cargo test --release --test world ;;
    agent) step "ruai agent against CPython" cargo test --release --test agent ;;
    site)
      step "build the website" ./run.sh build
      step "the website's py0 examples" ./run.sh test
      ;;
    check)
      step "no features" cargo check --lib --no-default-features
      for f in world pyrt py agent plot pyrt-python; do
        step "feature $f alone" cargo check --no-default-features --features "$f"
      done
      step "webgpu for wasm32" cargo check --lib --target wasm32-unknown-unknown --no-default-features --features webgpu
      step "py0 compiler for wasm32" cargo build --release --example py0_wasm --target wasm32-unknown-unknown --no-default-features --features py
      step "browser demo for wasm32" cargo build --release --target wasm32-unknown-unknown --manifest-path examples/webgpu-browser/Cargo.toml
      step "site crate for wasm32" cargo build --release --target wasm32-unknown-unknown --manifest-path site/wasm/Cargo.toml
      step "cuda builds (no GPU needed)" cargo check --all-targets --features cuda
      ;;
    lint)
      step "clippy" cargo clippy --all-targets -- -D warnings
      step "clippy webgpu" cargo clippy --all-targets --features webgpu -- -D warnings
      step "clippy cuda" cargo clippy --all-targets --features cuda -- -D warnings
      if [ "$OS" = Darwin ]; then
        step "clippy mps + accelerate" cargo clippy --all-targets --features mps,accelerate -- -D warnings
      fi
      step "rustdoc" env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features cuda
      ;;
    pretrained)
      if [ ! -d data/gpt2 ] || [ ! -d data/smollm2-135m ]; then
        skip "pretrained (run scripts/download_gpt2.sh and scripts/download_smollm.sh)"
        return
      fi
      step "real GPT-2 / SmolLM2 weights" cargo test --release --test inference --test tokenizer --test llama
      ;;
    ci)
      if [ "$OS" = Darwin ]; then
        step "cargo test with mps + accelerate" cargo test --release --features mps,accelerate
        group webgpu
        step "clippy mps + accelerate" cargo clippy --all-targets --features mps,accelerate -- -D warnings
        step "rustdoc mps" env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features mps
      else
        group cpu
        group pyo3
        group check
        group lint
      fi
      ;;
    all)
      group cpu
      group gpu
      group pyo3
      group site
      group check
      group lint
      group pretrained
      ;;
    deps)
      for t in cargo node "$PYTHON"; do
        if command -v "$t" >/dev/null 2>&1; then echo "ok       $t"; else echo "missing  $t"; fi
      done
      for m in numpy gymnasium sklearn matplotlib torch; do
        if "$PYTHON" -c "import $m" 2>/dev/null; then echo "ok       python: $m"; else echo "missing  python: $m (pip install ${m/sklearn/scikit-learn})"; fi
      done
      if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then echo "ok       wasm32 target"; else echo "missing  wasm32 target (./run.sh setup)"; fi
      if [ -d data/gpt2 ]; then echo "ok       data/gpt2"; else echo "missing  data/gpt2 (scripts/download_gpt2.sh; only for 'pretrained')"; fi
      ;;
    *)
      echo "unknown group: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [ $# -eq 0 ] || [ "$1" = help ] || [ "$1" = -h ] || [ "$1" = --help ]; then
  usage
  exit 0
fi

if [ "$1" = file ]; then
  [ $# -ge 2 ] || { echo "usage: ./test.sh file NAME [FILTER]" >&2; exit 2; }
  [ -f "tests/$2.rs" ] || { echo "no tests/$2.rs (have: $(ls tests/*.rs | xargs -n1 basename | sed 's/\.rs$//' | tr '\n' ' '))" >&2; exit 2; }
  step "tests/$2.rs${3:+ ($3)}" cargo test --release --test "$2" ${3:+"$3"}
else
  for g in "$@"; do
    [ "$g" = deps ] && { group deps; exit 0; }
    group "$g"
  done
fi

printf '\n\033[1m==> summary\033[0m\n'
printf '%s\n' "${RESULTS[@]}"
if [ "$FAILED" -gt 0 ]; then
  printf '%d step(s) failed\n' "$FAILED"
  exit 1
fi
