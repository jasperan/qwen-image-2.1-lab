"""Fast environment checks: no model load, no generation.

Run with the default pytest selection (these are not marked slow).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_torch_sees_a_cuda_device() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False; the GPU is not usable"
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_name(0)


def test_vllm_and_vllm_omni_import() -> None:
    vllm = importlib.import_module("vllm")
    omni = importlib.import_module("vllm_omni")
    assert vllm.__version__.startswith("0.29"), f"unexpected vllm version {vllm.__version__}"
    assert omni.__version__


def test_qwen_image_21_pipeline_is_available() -> None:
    from vllm_omni.diffusion.models.qwen_image_21 import QwenImage21Pipeline

    assert QwenImage21Pipeline.__name__ == "QwenImage21Pipeline"


def test_runtime_helpers_import_and_resolve() -> None:
    from qwen_lab.runtime import QwenImage21, resolve_model

    assert QwenImage21
    assert resolve_model()


def test_local_weights_if_present_look_complete() -> None:
    model_dir = REPO_ROOT / "models" / "Qwen-Image-2.1"
    if not model_dir.is_dir():
        pytest.skip("weights not downloaded yet (run scripts/download_model.sh)")
    index = model_dir / "model_index.json"
    assert index.is_file(), "model_index.json missing from the local weights checkout"
    payload = json.loads(index.read_text())
    assert (model_dir / "transformer").is_dir()
    assert (model_dir / "text_encoder").is_dir()
    assert (model_dir / "vae").is_dir()
    assert payload.get("_class_name")
