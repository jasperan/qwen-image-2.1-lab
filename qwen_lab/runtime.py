"""Load Qwen-Image-2.1 once and serve text-to-image and edit requests on vLLM-Omni.

Mirrors the offline examples that ship with vllm-omni (PR #7759):
``examples/offline_inference/text_to_image/text_to_image.py`` and
``examples/offline_inference/image_to_image/image_edit.py``. Keeping a single
``Omni`` instance alive lets a batch of prompts pay the model-load cost once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = REPO_ROOT / "models" / "Qwen-Image-2.1"
DEFAULT_MODEL = "Qwen/Qwen-Image-2.1"
DEFAULT_STEPS = 40
DEFAULT_CFG_SCALE = 1.0

# CPU-offload presets for diffusion_offload_config.
# "module" mode swaps whole components but on CUDA keeps DiT device storage
# resident between swaps (persistent staging for CUDA-graph pointer stability),
# so it only fits one generation on a 24 GB card. "layer" mode streams DiT
# blocks from pinned host memory and is the working choice there.
OFFLOAD_PRESETS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "dit": {"mode": "module", "components": ["dit"]},
    "dit+text_encoder": {"mode": "module", "components": ["dit", "text_encoder"]},
    "layer-dit": {"mode": "layer", "components": ["dit"]},
    "layer-dit+text_encoder": {"mode": "layer", "components": ["dit", "text_encoder"]},
}


def resolve_model(model: str | Path | None = None) -> str:
    """Prefer the local weights checkout; fall back to the Hugging Face repo id."""
    if model:
        return str(model)
    if LOCAL_MODEL_DIR.is_dir():
        return str(LOCAL_MODEL_DIR)
    return DEFAULT_MODEL


def to_pil_images(images: Sequence[Any]) -> list[Image.Image]:
    """Normalize VAE outputs (numpy arrays or PIL) to PIL images."""
    import numpy as np
    from diffusers.utils import numpy_to_pil

    normalized: list[Image.Image] = []
    for image in images:
        if isinstance(image, np.ndarray):
            normalized.extend(numpy_to_pil(image))
        else:
            normalized.append(image)
    return normalized


def peak_gpu_gib() -> float:
    """Peak torch-allocator usage on the current device, in GiB (0 without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        return 0.0


@dataclass
class Generation:
    """One generation round: decoded images plus wall-clock time."""

    images: list[Image.Image]
    seconds: float


class QwenImage21:
    """Thin single-process wrapper around vLLM-Omni's QwenImage21Pipeline."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        mode: str | None = None,
        offload: str | dict[str, Any] | None = "layer-dit",
        vae_tiling: bool = False,
        quantization_config: dict[str, Any] | None = None,
        enforce_eager: bool | None = None,
        init_timeout: int = 1800,
    ) -> None:
        """Build the pipeline.

        Args:
            mode: passed through to ``Omni`` (the upstream T2I example uses
                ``"text-to-image"``; the edit example passes nothing).
            offload: key of ``OFFLOAD_PRESETS`` or an explicit
                ``diffusion_offload_config`` mapping; ``None``/``"none"`` keeps
                every component on the GPU.
            quantization_config: per-component FP8 config, e.g.
                ``{"text_encoder": {"method": "fp8"}}`` (frees ~7 GiB of
                device memory for the Qwen3-VL encoder).
        """
        from vllm_omni.entrypoints.omni import Omni

        self.model = resolve_model(model)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "vae_use_tiling": vae_tiling,
            "init_timeout": init_timeout,
            "stage_init_timeout": init_timeout,
        }
        if isinstance(offload, dict):
            kwargs["diffusion_offload_config"] = offload
        elif offload not in OFFLOAD_PRESETS:
            raise ValueError(
                f"unknown offload preset {offload!r}; use one of {sorted(OFFLOAD_PRESETS)} or a config mapping"
            )
        elif OFFLOAD_PRESETS[offload] is not None:
            kwargs["diffusion_offload_config"] = OFFLOAD_PRESETS[offload]
        if mode is not None:
            kwargs["mode"] = mode
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if enforce_eager is not None:
            kwargs["enforce_eager"] = enforce_eager
        self.offload = offload
        self._omni = Omni(**kwargs)

    # Internal plumbing -------------------------------------------------

    def _sampling_params_list(
        self,
        *,
        width: int | None,
        height: int | None,
        steps: int,
        cfg_scale: float,
        seed: int,
        num_outputs: int,
    ) -> list[Any]:
        import torch
        from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        generator = torch.Generator(device=current_omni_platform.device_type)
        generator.manual_seed(seed)
        params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            seed=seed,
            generator=generator,
            true_cfg_scale=cfg_scale,
            num_inference_steps=steps,
            num_outputs_per_prompt=num_outputs,
        )

        defaults = list(getattr(self._omni, "default_sampling_params_list", None) or [])
        params_list = [clone_sampling_params(item) for item in defaults]
        if not params_list:
            return [params]
        replaced = False
        for index, item in enumerate(params_list):
            if isinstance(item, OmniDiffusionSamplingParams):
                params_list[index] = params
                replaced = True
        if not replaced and len(params_list) == 1:
            params_list = [params]
        return params_list

    def _generate(
        self,
        prompt_dict: dict[str, Any],
        *,
        width: int | None,
        height: int | None,
        steps: int,
        cfg_scale: float,
        seed: int,
        num_outputs: int,
    ) -> Generation:
        from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

        started = time.perf_counter()
        outputs = self._omni.generate(
            prompt_dict,
            sampling_params_list=self._sampling_params_list(
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                num_outputs=num_outputs,
            ),
        )
        seconds = time.perf_counter() - started

        images: list[Any] | None = None
        for output in outputs or []:
            images = getattr(output, "images", None)
            if images:
                break
        if not images and outputs:
            images = extract_images_from_outputs(outputs)
        if not images:
            raise RuntimeError(f"{self.model} returned no images")
        return Generation(images=to_pil_images(images), seconds=seconds)

    # Public API --------------------------------------------------------

    def text_to_image(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: int = 42,
        negative_prompt: str | None = None,
        num_outputs: int = 1,
    ) -> Generation:
        """Generate from text only (sizes must be multiples of 32)."""
        prompt_dict: dict[str, Any] = {"prompt": prompt, "modalities": ["image"]}
        if negative_prompt is not None:
            prompt_dict["negative_prompt"] = negative_prompt
        return self._generate(
            prompt_dict,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            num_outputs=num_outputs,
        )

    def edit(
        self,
        prompt: str,
        images: Image.Image | Sequence[Image.Image],
        *,
        width: int | None = None,
        height: int | None = None,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        seed: int = 42,
        negative_prompt: str | None = None,
        num_outputs: int = 1,
    ) -> Generation:
        """Edit or combine 1-4 reference images; size defaults to the last image's ratio."""
        payload: Any = images if isinstance(images, Image.Image) else list(images)
        prompt_dict: dict[str, Any] = {"prompt": prompt, "multi_modal_data": {"image": payload}}
        if negative_prompt is not None:
            prompt_dict["negative_prompt"] = negative_prompt
        return self._generate(
            prompt_dict,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            num_outputs=num_outputs,
        )
