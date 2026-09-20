# Qwen-Image-2.1 on vLLM-Omni: Test Report

**Date:** 2026-09-20
**Host:** single NVIDIA A10 (23028 MiB, 22.06 GiB usable), driver 595.71.05
**Stack:** vllm 0.29.0 · vllm-omni 0.29.0rc2.dev17+g3ea9e6050 (PR [#7759](https://github.com/vllm-project/vllm-omni/pull/7759) head `3ea9e605011fc2a0bfa9f3e8531286fb683068dd`) · torch 2.13.0+cu130 · transformers 5.14.1 · diffusers 0.40.0
**Weights:** `Qwen/Qwen-Image-2.1`, 31 GiB local checkout (bf16 DiT 7.1B + Qwen3-VL-8B text encoder + 16x RGBA autoencoder)
**Report status:** PASS with documented resource limits

---

## Executive summary

| Metric | Value |
| --- | --- |
| Text-to-image gallery | 8 / 8 images at 40 steps, 1024-class sizes |
| Single-reference edits | 2 / 2 demos (recolor, local text replacement) |
| Multi-reference edits (2 images) | 0 / 5 attempts - exceeds 22.06 GiB device memory at every configuration tried |
| Fast test suite | 5 passed, 0 failed (13.4 s) |
| Slow GPU suite | 3 passed, 0 failed (42.4 s) |
| Server API suite | 3 passed, 0 failed (43.8 s incl. 15.6 s server init) |
| Steady-state speed | ~48 s/image (1024-class, 40 steps), ~31 s/image (768x768, 40 steps) |
| Peak device memory | 20.3 GiB observed; 19.3 GiB torch allocator peak |
| Reproduce | `bash scripts/setup.sh && bash scripts/download_model.sh && make gallery` |

Quality highlights: headline typography is exact (poster and app-mockup text rendered character-perfect), portraits and
landscapes are photorealistic, and the RGBA autoencoder produced a real alpha channel (76.6 % fully transparent
pixels for the cutout prompt). Sub-headline body text (< ~20 px) degrades into pseudo-glyphs, which is the one
quality caveat worth planning around.

---

## Test inventory and results

### Fast suite (default selection)

```bash
.venv/bin/python -m pytest -q        # 5 passed, 6 deselected in 13.36s
```

Covers: CUDA visibility, vllm/vllm-omni import + versions, `QwenImage21Pipeline` availability, runtime helper
imports, local weight layout.

### Slow suite (loads the pipeline on the GPU)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m pytest -m slow -v
# 3 passed, 8 deselected in 42.35s
```

| Test | What it proves |
| --- | --- |
| `test_text_to_image_returns_a_valid_png` | 512x512 at 6 steps returns a PNG with real content (stddev > 2) |
| `test_transparent_prompt_yields_declared_channels` | the cutout prompt yields RGBA with a genuinely transparent alpha channel |
| `test_edit_replaces_reference_color` | a synthetic checkerboard reference is repainted (reference red removed) |

One pipeline instance serves all three tests; initializing a second `Omni` in the same process fails with
`model parallel state must be empty before initialization`, so the suite shares one session-scoped pipeline.

### Server suite (OpenAI-compatible API)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m pytest -m server -v
# 3 passed, 8 deselected in 43.79s
```

Starts `vllm serve ... --omni --diffusion-offload-config '{"mode":"layer","components":["dit"]}' --vae-use-tiling`
in a subprocess (server initialized in 15.58 s, layer-wise offloading on 32 blocks), then:

| Endpoint | Result |
| --- | --- |
| `GET /health` | 200 |
| `POST /v1/images/generations` (512x512, 6 steps) | 200, base64 PNG, non-uniform content |
| `POST /v1/images/edits` (multipart, 1 reference) | 200, base64 PNG, requested size |

Server log: `outputs/server.log`.

---

## Gallery results (text-to-image, 40 steps, seed 42, true_cfg_scale 1.0)

| id | size | mode | time | quality notes |
| --- | --- | --- | --- | --- |
| product-teapot | 1024x1024 | RGBA | 48.4 s | photoreal glaze and clay texture |
| poster-typography | 832x1216 | RGBA | 47.5 s | "VISIT MARS" and "Gravity Optional" exact |
| infographic-blackholes | 1216x832 | RGBA | 47.7 s | title + 4 panel headings exact; bullet body text garbles |
| panorama-fjord | 1536x640 | RGBA | 46.6 s | coherent wide composition, consistent reflection |
| portrait-fisherman | 1024x1024 | RGBA | 48.8 s | convincing skin/beard micro-texture |
| transparent-sneaker | 1024x1024 | RGBA | 49.0 s | 76.6 % of pixels fully transparent, clean edges |
| ui-mockup | 832x1216 | RGBA | 48.5 s | headline, `$18 / bag`, `SUBSCRIBE` exact; body + tab labels degrade |
| dragonfly-macro | 1024x1024 | RGBA | 49.0 s | high-frequency detail holds up |

Total 385.6 s (48.2 s/image average) in a single process. All outputs are RGBA because the autoencoder is
4-channel in/out; only transparency-prompts actually use alpha, otherwise alpha sits at 190-255 (light noise).

Artifacts: `outputs/gallery/*.png`, `outputs/gallery/montage.png`, `outputs/gallery/gallery.html`,
`outputs/gallery/manifest.json`. Committed, downscaled copies used by the README live in `docs/images/`.

## Edit results (40 steps, seed 42)

| demo | input | size | result |
| --- | --- | --- | --- |
| Glaze recolor | teapot photo | 768x768 | indigo -> crimson; table, lighting and composition preserved (30.6 s) |
| App text edit | coffee UI mockup | 768x768 | `$18 / bag` -> `$24 / bag` and `SUBSCRIBE` -> `PREORDER`, exact, rest untouched (30.7 s) |
| Two-reference combine | teapot + sneaker | 1024/768/640/512 | **OOM at every size** - see below |

`outputs/edits/teapot-crimson.png`, `outputs/edits/ui-price-change.png`.

---

## Resource findings (what actually fits on 22.06 GiB)

1. **Module-level offload (`--enable-cpu-offload` / `{"mode":"module"} `) does not work on a 24 GB CUDA card.**
   On CUDA the sequential backend stages DiT weights with `persistent_dit_staging=True`, and `PinnedModuleStager`
   keeps the device storages resident while "offloaded" (they exist to keep CUDA-graph weight pointers stable).
   Consequences observed:
   - `components=["dit"]`: OOM during the init dummy run - the 17.5 GiB text encoder stays resident when the DiT runs.
   - `components=["dit","text_encoder"]`: init passes, first generation passes, the **second** generation OOMs when
     the text encoder is swapped back in alongside the retained DiT storage (21.62 GiB allocated, 23 MiB free).
   Source: `vendor/vllm-omni/vllm_omni/diffusion/offloader/sequential_backend.py` (call site ~line 419) and
   `offloader/module_residency.py` (`offload()` + `retain_device_storage`).
2. **Layer-wise offload (`{"mode":"layer","components":["dit"]}`) is the working configuration** on this card:
   DiT blocks stream from pinned host memory (32 blocks reported), the text encoder stays resident. Repeat
   generations in one process work; peak allocator memory 19.3 GiB, device peak 20.3 GiB, ~1.2 s/step at 1024x1024.
   CUDA-graph decode is auto-disabled in this mode (`transformer blocks carry offload/cache hooks`); decode runs eager.
3. **FP8 text encoder is not reachable on 22 GiB**: `quantization_config={"text_encoder": {"method": "fp8"}}` is
   accepted (per-component quantization is wired up), but online quantization needs the bf16 weights plus the fp8
   copy at once and OOMs during init (21.77 GiB allocated before the dummy run finishes).
4. **Multi-reference edits are encode-bound, not size-bound.** Condition images are always resized to ~1024x1024
   area before the Qwen3-VL encode (`OUTPUT_RESOLUTION = 1024` in the pipeline), and two references push the
   encode/denoise path past 22 GiB regardless of the requested output size (tested 1024, 768, 640, 512, plus the
   fp8-encoder attempt). Single-reference edits fit up to 768x768 output.
5. **One pipeline per process**: a second `Omni` in the same process fails to initialize (model-parallel state is
   not torn down between instances).
6. **Init cost is small**: ~19 s offline (layer-wise leaves DiT weights on host) and 15.6 s for the API server;
   the first request also pays prefix-cache/graph setup (~1.3x per upstream).

---

## Quality observations

- **Typography**: headline-size text (>= ~40 px) is exact - both the poster and the app mockup rendered long quoted
  strings character-perfect, which is the model's signature strength. Body copy and small labels degrade into
  plausible pseudo-glyphs; keep prompts to few, large words when the text must be real.
- **Photorealism**: portrait skin, wood grain, glaze pooling and macro insect detail all hold up at 1024-class sizes.
- **Transparency**: the pipeline returns RGBA for every prompt; a cutout prompt produced 76.6 % fully transparent
  pixels with clean edges, so the 16x RGBA autoencoder works end to end on this path.
- **Editing fidelity**: both edit demos preserved everything outside the requested change (table/lighting for the
  teapot, layout/chrome for the mockup) while applying the requested color/text change.
- **Panorama**: 1536x640 stayed coherent across the frame (no duplicated peaks, consistent water reflection).

## Recommendations

1. Use `--offload layer-dit` (the repo default) plus `--vae-use-tiling` on 24 GB cards; use `--offload none` on
   >= 40 GB cards to keep full-speed decode.
2. Keep edit outputs at <= 768x768 and one reference image on this hardware; run multi-reference edits on a
   >= 32 GB GPU or with an offline-quantized encoder.
3. Report the module-mode offload behavior upstream: retaining device storages on CUDA makes the compat alias
   unusable for memory-bound cards and turns repeat requests into OOM after the first success.
4. For production text rendering, keep wording short and large; verify any critical glyphs visually before shipping.
5. Consider an offline (pre-quantized) FP8 text encoder checkpoint if 24 GB cards need more headroom.

## Reproduce

```bash
bash scripts/setup.sh
bash scripts/download_model.sh
make gallery
make edit
make test && make test-gpu && make test-server
```

Environment knobs used for every GPU run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`LAB_OFFLOAD=layer-dit` (default), `--vae-use-tiling` on.
