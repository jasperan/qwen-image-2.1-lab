#!/usr/bin/env bash
# Serve Qwen-Image-2.1 behind the vLLM-Omni OpenAI-compatible API.
#
#   POST /v1/images/generations   (JSON; image at .data[0].b64_json)
#   POST /v1/images/edits         (multipart form; repeat -F image=@file)
#   POST /v1/chat/completions     (image_url content parts)
#
# On a 24 GB card the default flags stream DiT blocks from host memory and
# enable VAE tiling so the checkpoint fits. Module-level offload
# (--enable-cpu-offload) does NOT fit here: on CUDA it retains DiT device
# storage between swaps, so the second generation OOMs. Override via EXTRA_ARGS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8091}"

if [ -n "${MODEL:-}" ]; then
  MODEL_ARG="$MODEL"
elif [ -d "$ROOT/models/Qwen-Image-2.1" ]; then
  MODEL_ARG="$ROOT/models/Qwen-Image-2.1"
else
  MODEL_ARG="Qwen/Qwen-Image-2.1"
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA <<< "$EXTRA_ARGS"
else
  EXTRA=(--diffusion-offload-config '{"mode":"layer","components":["dit"]}' --vae-use-tiling)
fi

echo "serving $MODEL_ARG on port $PORT (extra args: ${EXTRA[*]})"
exec "$ROOT/.venv/bin/vllm" serve "$MODEL_ARG" --omni --port "$PORT" "${EXTRA[@]}"
