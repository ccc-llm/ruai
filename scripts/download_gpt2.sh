#!/usr/bin/env bash
# Download GPT-2 small (124M) weights and tokenizer from Hugging Face
# (openai-community/gpt2, MIT license) into data/gpt2/.
# Set MODEL=gpt2-medium / gpt2-large / gpt2-xl for the larger sizes.
set -euo pipefail
MODEL="${MODEL:-gpt2}"
DIR="${DIR:-data/$MODEL}"
BASE="https://huggingface.co/openai-community/$MODEL/resolve/main"
mkdir -p "$DIR"
for f in config.json vocab.json merges.txt tokenizer.json model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "have $DIR/$f"
    else
        echo "downloading $f ..."
        curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
        mv "$DIR/$f.part" "$DIR/$f"
    fi
done
ls -lh "$DIR"
