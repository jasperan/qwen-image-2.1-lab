"""Shared pytest fixtures for the Qwen-Image-2.1 lab.

Environment knobs:
    LAB_MODEL        model id or local path (default: models/Qwen-Image-2.1)
    LAB_OFFLOAD      layer-dit | layer-dit+text_encoder | dit | dit+text_encoder | none
    LAB_VAE_TILING   1 to enable VAE tiling (default: 0)
    LAB_INIT_TIMEOUT pipeline init timeout in seconds (default: 1800)
"""

from __future__ import annotations

import os

import pytest

from qwen_lab.runtime import QwenImage21


def model_arg() -> str | None:
    return os.environ.get("LAB_MODEL") or None


def offload_arg() -> str | None:
    value = os.environ.get("LAB_OFFLOAD", "layer-dit")
    return value or None


def vae_tiling_arg() -> bool:
    return os.environ.get("LAB_VAE_TILING", "0") == "1"


def init_timeout_arg() -> int:
    return int(os.environ.get("LAB_INIT_TIMEOUT", "1800"))


@pytest.fixture(scope="session")
def lab() -> QwenImage21:
    """One pipeline per process; it serves both text-to-image and edits.

    A second ``Omni`` cannot be initialized in the same process (the diffusion
    worker keeps model-parallel state alive), so every slow test shares this
    instance. ``mode`` is left unset, matching the upstream edit example.
    """
    return QwenImage21(
        model_arg(),
        mode=None,
        offload=offload_arg(),
        vae_tiling=vae_tiling_arg(),
        init_timeout=init_timeout_arg(),
    )
