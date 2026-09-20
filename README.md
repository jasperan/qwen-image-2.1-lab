# qwen-image-2.1-lab

A reproducible testbed for **Qwen-Image-2.1** (7.1B unified text-to-image and editing model with native
RGBA output) served by **vLLM-Omni**.

Upstream support is not merged yet, so this repo pins the whole stack:
`vllm==0.29.0` plus a vendored checkout of
[vllm-omni PR #7759](https://github.com/vllm-project/vllm-omni/pull/7759) (`qwen-image-2.1` branch,
commit `3ea9e605` at the time of writing). `scripts/setup.sh` re-pins that checkout on every run and
writes the resolved SHA to `vendor/vllm-omni.revision`.

Upstream recipe: <https://recipes.vllm.ai/Qwen/Qwen-Image-2.1>

## What is in here

| Path | Purpose |
| --- | --- |
| `scripts/setup.sh` | uv venv + vLLM 0.29.0 + editable vllm-omni checkout |
| `scripts/download_model.sh` | pull the ~33 GB checkpoint into `models/Qwen-Image-2.1` |
| `scripts/serve.sh` | `vllm serve ... --omni` OpenAI-compatible server |
| `configs/prompts.json` | prompt battery: typography, infographic, panorama, portrait, transparency, UI, macro |
| `examples/generate_gallery.py` | batch text-to-image runs, PNGs + `manifest.json` + `montage.png` + `gallery.html` |
| `examples/edit_image.py` | edit or combine 1-4 reference images |
| `qwen_lab/runtime.py` | one `Omni` instance, many requests; offload presets |
| `tests/` | fast env checks (default), offline pipeline smoke tests (`-m slow`), server API tests (`-m server`) |
| `outputs/` | generated images, manifests, server log (gitignored) |
| `REPORT.md` | actual run report: timings, memory findings, quality notes, limitations |

## Measured on this box

| Run | Result |
| --- | --- |
| 8-prompt gallery, 40 steps, 1024-class | 8/8 images, ~48 s each, peak 20.3 GiB |
| edits, 40 steps, 768x768 | recolor + local text replacement, ~31 s each |
| slow GPU suite (`-m slow`) | 3 passed in 42 s (one pipeline load + 3 generations) |
| server suite (`-m server`) | 3 passed in 44 s (server init 15.6 s) + `outputs/server.log` |

Quality: headline typography is character-exact, portraits/landscapes are photorealistic, cutout prompts give
real transparency (76.6 % transparent pixels measured). Body text below ~20 px turns into pseudo-glyphs.

## Hardware notes

Measured on this box (single NVIDIA A10, 22.06 GiB usable): the bf16 checkpoint does not fit whole, and
**module-level CPU offload does not save memory on CUDA** - the sequential backend keeps DiT device storage
resident between swaps (persistent staging for CUDA-graph pointer stability), so the second generation OOMs.
What works, and is the default here:

- `--offload layer-dit` -> `diffusion_offload_config={"mode":"layer","components":["dit"]}`: DiT blocks stream
  from pinned host memory (32 blocks), the Qwen3-VL encoder stays resident. Peak observed: 20.3 GiB device.
- `--vae-use-tiling` on top, especially for edits.
- `--offload none` on >= 40 GB cards (also restores CUDA-graph decode, which layer hooks disable).

Known limits on 24 GB:

- single-reference edits fit up to 768x768 output (the edit example default); 1024x1024 edits OOM
- multi-reference edits (2+ images) OOM at every size - condition images are always encoded at ~1024x1024 area,
  so the encode phase needs > 22 GiB
- online FP8 quantization of the text encoder OOMs during init (bf16 + fp8 weights coexist briefly)

One `Omni` pipeline per process: a second instance cannot initialize in the same process, so the test suite
shares one session-scoped pipeline for both text-to-image and edits.

Full evidence: [REPORT.md](REPORT.md).

## Quickstart

```bash
bash scripts/setup.sh            # creates .venv and vendor/vllm-omni
bash scripts/download_model.sh   # ~33 GB into models/Qwen-Image-2.1
make gallery                     # 8 prompts, 40 steps, 1024-class sizes
make edit                        # single-reference edit demo
```

Direct commands:

```bash
.venv/bin/python examples/generate_gallery.py --config configs/prompts.json --only poster-typography
.venv/bin/python examples/edit_image.py \
    --image outputs/gallery/00-product-teapot.png \
    --prompt "Change the glaze to a deep crimson red, keep the same photo style" \
    --out outputs/edits/teapot-crimson.png
```

Serving:

```bash
bash scripts/serve.sh            # port 8091; layer DiT offload + VAE tiling by default

curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "'"$PWD"'/models/Qwen-Image-2.1", "prompt": "A ceramic teapot on a wooden table",
       "size": "1024x1024", "num_inference_steps": 40, "true_cfg_scale": 1.0, "seed": 42}'
```

`/v1/images/edits` is **multipart/form-data**, not JSON, and takes up to four image parts:

```bash
curl -X POST http://localhost:8091/v1/images/edits \
  -F image=@plate.png \
  -F 'prompt=Write the words "FRESH BASIL" on this plate in dark green lettering' \
  -F model="$PWD/models/Qwen-Image-2.1" -F size=1024x1024 \
  -F num_inference_steps=40 -F true_cfg_scale=1.0 -F seed=42
```

## Tests

```bash
make test          # fast: env only, no generation
make env-report    # same, verbose
make test-gpu      # slow: loads the pipeline, generates at 512px / 6 steps
make test-server   # starts the server, calls /v1/images/generations and /v1/images/edits
```

Environment knobs for the GPU suites: `LAB_MODEL`, `LAB_OFFLOAD` (`layer-dit`, `layer-dit+text_encoder`,
`dit`, `dit+text_encoder`, `none`), `LAB_VAE_TILING=1`, `LAB_INIT_TIMEOUT`, `LAB_SERVER_TIMEOUT`.

## Sampling defaults (from the upstream recipe)

- 40 denoising steps, `true_cfg_scale=1.0` (no negative prompt, so the CFG branch stays disabled)
- sizes are floored to multiples of 32; 1024x1024 class is the checkpoint's native area
- the server's own default is 50 steps, so always send `num_inference_steps`
- up to 4 reference images per request on this path; a fifth is rejected (2+ references need more than
  22 GiB on this card, see Hardware notes)

## Caveats

- vllm-omni landing support for this model is unmerged; the pinned SHA is the tested one
- `/v1/images/edits` rejects JSON bodies (connection reset) - always send multipart form data
- model license is `qwen-research`; check the model card before any non-research use
