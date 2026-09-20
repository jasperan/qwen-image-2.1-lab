"""Slow smoke tests that load the pipeline and generate on the GPU.

Run with:
    .venv/bin/python -m pytest -m slow -v

The point is structural correctness (request path, output decode, alpha
channel), not image quality; steps are kept low to bound runtime.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageStat

pytestmark = pytest.mark.slow

SMOKE_STEPS = 6
SMOKE_SIZE = 512


def assert_looks_like_an_image(path, size):
    reloaded = Image.open(path)
    assert reloaded.size == size, f"expected {size}, got {reloaded.size}"
    grayscale = reloaded.convert("L")
    stddev = ImageStat.Stat(grayscale).stddev[0]
    assert stddev > 2.0, f"image looks uniform (stddev {stddev:.2f}); the pipeline likely failed silently"
    return reloaded


def test_text_to_image_returns_a_valid_png(lab, tmp_path) -> None:
    generation = lab.text_to_image(
        "A single ripe red apple on a plain white table",
        width=SMOKE_SIZE,
        height=SMOKE_SIZE,
        steps=SMOKE_STEPS,
        seed=7,
    )
    assert generation.seconds > 0
    assert generation.images, "no images returned"
    path = tmp_path / "t2i-smoke.png"
    generation.images[0].save(path)
    assert_looks_like_an_image(path, (SMOKE_SIZE, SMOKE_SIZE))


def test_transparent_prompt_yields_declared_channels(lab, tmp_path) -> None:
    """The 16x RGBA autoencoder should be able to emit transparency on request."""
    generation = lab.text_to_image(
        "A glossy red ceramic mug, studio product cutout, isolated on a fully transparent background, no background",
        width=SMOKE_SIZE,
        height=SMOKE_SIZE,
        steps=SMOKE_STEPS,
        seed=11,
    )
    image = generation.images[0]
    path = tmp_path / "rgba-smoke.png"
    image.save(path)
    reloaded = Image.open(path)
    assert reloaded.mode in {"RGB", "RGBA"}, f"unexpected mode {reloaded.mode}"
    assert_looks_like_an_image(path, (SMOKE_SIZE, SMOKE_SIZE))
    if reloaded.mode == "RGBA":
        alpha = reloaded.getchannel("A")
        assert alpha.getextrema()[0] < 250, "alpha channel is fully opaque despite the transparent-background prompt"
    else:
        pytest.fail("model returned RGB for a transparent-background prompt; transparency is not reachable on this path")


def test_edit_replaces_reference_color(lab, tmp_path) -> None:
    reference = Image.new("RGBA", (SMOKE_SIZE, SMOKE_SIZE), (240, 240, 240, 255))
    draw = ImageDraw.Draw(reference)
    for x in range(0, SMOKE_SIZE, 64):
        for y in range(0, SMOKE_SIZE, 64):
            if (x // 64 + y // 64) % 2 == 0:
                draw.rectangle([x, y, x + 31, y + 31], fill=(200, 30, 30, 255))

    generation = lab.edit(
        "Repaint the red squares as deep blue, keep the white background",
        reference,
        width=SMOKE_SIZE,
        height=SMOKE_SIZE,
        steps=SMOKE_STEPS,
        seed=13,
    )
    assert generation.images
    path = tmp_path / "edit-smoke.png"
    generation.images[0].save(path)
    reloaded = assert_looks_like_an_image(path, (SMOKE_SIZE, SMOKE_SIZE))
    red_pixels = sum(1 for pixel in reloaded.convert("RGB").getdata() if pixel[0] > 120 and pixel[1] < 100 and pixel[2] < 100)
    assert red_pixels < SMOKE_SIZE * SMOKE_SIZE * 0.1, "reference red survived; edit request was ignored"
