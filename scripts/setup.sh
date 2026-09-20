#!/usr/bin/env bash
# Create .venv and install vLLM + vllm-omni at the Qwen-Image-2.1 branch.
#
# Support for Qwen-Image-2.1 lives in vllm-omni PR #7759, which is not merged
# upstream yet, so the vendored checkout is pinned to the PR head.
#
# Note: this script intentionally never calls `source .venv/bin/activate`; it
# drives uv and the venv by absolute path so it works from any shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VLLM_OMNI_PR="${VLLM_OMNI_PR:-7759}"
VLLM_OMNI_REF="${VLLM_OMNI_REF:-qwen-image-2.1}"
VLLM_OMNI_DIR="$ROOT/vendor/vllm-omni"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}
mkdir -p "$ROOT/vendor"

if [ ! -d "$VLLM_OMNI_DIR/.git" ]; then
  git clone "$VLLM_OMNI_REPO" "$VLLM_OMNI_DIR"
fi
git -C "$VLLM_OMNI_DIR" fetch origin "pull/${VLLM_OMNI_PR}/head"
git -C "$VLLM_OMNI_DIR" checkout -B "$VLLM_OMNI_REF" FETCH_HEAD
git -C "$VLLM_OMNI_DIR" rev-parse HEAD | tee "$ROOT/vendor/vllm-omni.revision"

cd "$ROOT"
[ -d .venv ] || uv venv --python 3.12 --seed
export VIRTUAL_ENV="$ROOT/.venv"
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
uv pip install -e "$VLLM_OMNI_DIR"
uv pip install pytest pytest-asyncio requests pillow huggingface_hub hf_xet

"$ROOT/.venv/bin/python" - <<'PY'
import torch
import vllm
import vllm_omni

print("vllm:", vllm.__version__)
print("vllm-omni:", vllm_omni.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete."
echo "Next: bash scripts/download_model.sh"
