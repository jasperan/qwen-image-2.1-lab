#!/usr/bin/env python
"""Edit or combine reference images with Qwen-Image-2.1 (vLLM-Omni).

Up to four reference images can be supplied; the prompt can address them by
order ("the second image") because each is tagged <image1>, <image2>, ...

Examples:
    .venv/bin/python examples/edit_image.py \
        --image outputs/gallery/00-product-teapot.png \
        --prompt 'Change the glaze to a deep crimson red' \
        --out outputs/edits/teapot-red.png

    .venv/bin/python examples/edit_image.py \
        --image a.png b.png \
        --prompt 'Combine these images into a single scene' \
        --out outputs/edits/combined.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_lab.runtime import DEFAULT_CFG_SCALE, DEFAULT_STEPS, OFFLOAD_PRESETS, QwenImage21, resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True, help="1-4 reference images.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--model", default=None)
    # 768 keeps single-reference edits inside 22 GiB on a 24 GB card: the Qwen3-VL
    # encoder stays resident while condition images are encoded at ~1024x1024 area,
    # and 1024x1024 edit outputs OOM on an A10. Multi-reference edits need more.
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--color-format",
        default="RGBA",
        choices=["RGB", "RGBA"],
        help="How reference images are decoded; RGBA keeps transparency.",
    )
    parser.add_argument(
        "--offload",
        default="layer-dit",
        choices=sorted(OFFLOAD_PRESETS),
        help="CPU-offload strategy; 'layer-dit' streams DiT blocks on 24 GB cards.",
    )
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument(
        "--fp8-text-encoder",
        action="store_true",
        help="Quantize the Qwen3-VL language model with FP8 (vision tower stays BF16); frees ~7 GiB.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.image) > 4:
        raise SystemExit("Qwen-Image-2.1 accepts at most 4 reference images on this path")
    if len(args.image) > 1:
        print(
            "note: multi-reference edits need more than 22 GiB of device memory; "
            "on a 24 GB card they OOM at any output size (see REPORT.md)",
        )

    images = []
    for path_str in args.image:
        path = Path(path_str)
        if not path.is_file():
            raise SystemExit(f"reference image not found: {path}")
        images.append(Image.open(path).convert(args.color_format))

    model = resolve_model(args.model)
    print(f"model: {model}")
    print(f"references: {len(images)} ({args.color_format}), offload: {args.offload}")

    lab = QwenImage21(
        model,
        mode=None,
        offload=args.offload,
        vae_tiling=args.vae_use_tiling,
        quantization_config={"text_encoder": {"method": "fp8"}} if args.fp8_text_encoder else None,
        enforce_eager=True if args.enforce_eager else None,
        init_timeout=args.timeout,
    )
    print("pipeline ready")

    generation = lab.edit(
        args.prompt,
        images,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    image = generation.images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved {out_path} [{image.mode} {image.width}x{image.height}] in {generation.seconds:.1f}s")


if __name__ == "__main__":
    main()
