#!/usr/bin/env bash
# Download a small Llama-architecture model with its tokenizer.json from
# Hugging Face into data/<name>/ (default: HuggingFaceTB/SmolLM2-135M,
# Apache-2.0, ~270 MB). Any LlamaForCausalLM repo with a single
# model.safetensors works, e.g. REPO=HuggingFaceTB/SmolLM2-360M.
set -euo pipefail
REPO="${REPO:-HuggingFaceTB/SmolLM2-135M}"
DIR="${DIR:-data/$(basename "$REPO" | tr '[:upper:]' '[:lower:]')}"
BASE="https://huggingface.co/$REPO/resolve/main"
mkdir -p "$DIR"
for f in config.json tokenizer.json model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "have $DIR/$f"
    else
        echo "downloading $f ..."
        curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
        mv "$DIR/$f.part" "$DIR/$f"
    fi
done
ls -lh "$DIR"
