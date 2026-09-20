#!/usr/bin/env bash
# Download the Qwen-Image-2.1 weights (~33 GB) into models/Qwen-Image-2.1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen-Image-2.1}"
DEST="${DEST:-$ROOT/models/Qwen-Image-2.1}"

if [ -x "$ROOT/.venv/bin/hf" ]; then
  HF="$ROOT/.venv/bin/hf"
else
  HF="$ROOT/.venv/bin/huggingface-cli"
fi

mkdir -p "$(dirname "$DEST")"
"$HF" download "$MODEL_ID" --local-dir "$DEST"
du -sh "$DEST"
echo "Weights ready: $DEST"
