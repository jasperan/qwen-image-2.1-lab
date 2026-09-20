#!/usr/bin/env python
"""Generate a text-to-image gallery for Qwen-Image-2.1 from a prompt config.

Example:
    .venv/bin/python examples/generate_gallery.py --config configs/prompts.json
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_lab.runtime import (
    DEFAULT_CFG_SCALE,
    DEFAULT_STEPS,
    OFFLOAD_PRESETS,
    QwenImage21,
    peak_gpu_gib,
    resolve_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prompts.json", help="Prompt battery JSON.")
    parser.add_argument("--outdir", default="outputs/gallery", help="Where images and the manifest land.")
    parser.add_argument("--model", default=None, help="Model id or local path (default: ./models/Qwen-Image-2.1).")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these prompt ids.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Denoising steps (checkpoint wants 40).")
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE, help="True CFG scale (1.0 = off).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offload",
        default="layer-dit",
        choices=sorted(OFFLOAD_PRESETS),
        help="CPU-offload strategy; 'layer-dit' streams DiT blocks on 24 GB cards.",
    )
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800, help="Pipeline init timeout in seconds.")
    parser.add_argument("--no-montage", dest="montage", action="store_false")
    return parser.parse_args()


def load_prompts(config_path: Path, only: list[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text())
    prompts = payload["prompts"] if isinstance(payload, dict) else payload
    if only:
        wanted = set(only)
        prompts = [item for item in prompts if item["id"] in wanted]
        missing = wanted - {item["id"] for item in prompts}
        if missing:
            raise SystemExit(f"unknown prompt id(s): {', '.join(sorted(missing))}")
    return prompts


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_contact_sheet(records: list[dict[str, Any]], outdir: Path, columns: int = 3) -> Path:
    cell_w, cell_h = 560, 640
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(20)
    small_font = load_font(16)
    for index, record in enumerate(records):
        col, row = index % columns, index // columns
        x0, y0 = col * cell_w, row * cell_h
        with Image.open(record["path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell_w - 40, cell_h - 110))
        sheet.paste(thumb, (x0 + (cell_w - thumb.width) // 2, y0 + 20))
        draw.text((x0 + 20, y0 + cell_h - 80), record["id"], fill=(240, 240, 240), font=title_font)
        draw.text(
            (x0 + 20, y0 + cell_h - 52),
            f"{record['width']}x{record['height']}  {record['seconds']:.1f}s  seed {record['seed']}",
            fill=(150, 150, 155),
            font=small_font,
        )
    path = outdir / "montage.png"
    sheet.save(path)
    return path


def build_html(records: list[dict[str, Any]], outdir: Path, meta: dict[str, Any]) -> Path:
    cards = []
    for record in records:
        name = html.escape(Path(record["path"]).name)
        cards.append(
            f"""
        <figure>
          <img src="{name}" alt="{html.escape(record['id'])}">
          <figcaption>
            <strong>{html.escape(record['id'])}</strong> &middot; {record['width']}x{record['height']}
            &middot; {record['seconds']:.1f}s &middot; seed {record['seed']}
            <p>{html.escape(record['prompt'])}</p>
          </figcaption>
        </figure>"""
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Qwen-Image-2.1 gallery</title>
<style>
  body {{ background:#111113; color:#e9e9ec; font:15px/1.5 system-ui, sans-serif; margin:0; padding:32px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  p.meta {{ color:#9a9aa2; margin:0 0 28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:24px; }}
  figure {{ margin:0; background:#1a1a1e; border:1px solid #2a2a30; border-radius:10px; overflow:hidden; }}
  figure img {{ width:100%; display:block; }}
  figcaption {{ padding:12px 14px; font-size:13px; color:#c9c9d1; }}
  figcaption p {{ color:#9a9aa2; margin:6px 0 0; }}
</style>
</head>
<body>
  <h1>Qwen-Image-2.1 &mdash; vLLM-Omni gallery</h1>
  <p class="meta">model {html.escape(str(meta['model']))} &middot; {meta['steps']} steps &middot; true cfg {meta['cfg_scale']} &middot; {len(records)} images</p>
  <div class="grid">{''.join(cards)}
  </div>
</body>
</html>
"""
    path = outdir / "gallery.html"
    path.write_text(page)
    return path


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.config), args.only)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = resolve_model(args.model)
    print(f"model: {model}")
    print(f"offload: {args.offload}, vae tiling: {args.vae_use_tiling}, prompts: {len(prompts)}")

    lab = QwenImage21(
        model,
        mode="text-to-image",
        offload=args.offload,
        vae_tiling=args.vae_use_tiling,
        enforce_eager=True if args.enforce_eager else None,
        init_timeout=args.timeout,
    )
    print("pipeline ready")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(prompts):
        seed = item.get("seed", args.seed)
        steps = item.get("steps", args.steps)
        width, height = item["width"], item["height"]
        print(f"[{index + 1}/{len(prompts)}] {item['id']} ({width}x{height}, {steps} steps)")
        started = time.perf_counter()
        generation = lab.text_to_image(
            item["prompt"],
            width=width,
            height=height,
            steps=steps,
            cfg_scale=args.cfg_scale,
            seed=seed,
        )
        image = generation.images[0]
        path = outdir / f"{index:02d}-{item['id']}.png"
        image.save(path)
        record = {
            "id": item["id"],
            "category": item.get("category"),
            "prompt": item["prompt"],
            "notes": item.get("notes"),
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "steps": steps,
            "cfg_scale": args.cfg_scale,
            "seed": seed,
            "seconds": generation.seconds,
            "peak_gpu_gib": round(peak_gpu_gib(), 2),
        }
        records.append(record)
        print(f"    saved {path} [{image.mode}] in {generation.seconds:.1f}s (peak alloc {peak_gpu_gib():.1f} GiB)")

    meta = {
        "model": model,
        "offload": args.offload,
        "vae_use_tiling": args.vae_use_tiling,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "entries": records,
    }
    (outdir / "manifest.json").write_text(json.dumps(meta, indent=2))
    if args.montage and records:
        print(f"montage: {build_contact_sheet(records, outdir)}")
    print(f"html:    {build_html(records, outdir, meta)}")
    print(f"manifest: {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
