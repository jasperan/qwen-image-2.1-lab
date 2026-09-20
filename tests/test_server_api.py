"""Smoke tests against the vLLM-Omni OpenAI-compatible server.

Run with:
    .venv/bin/python -m pytest -m server -v

Starts ``vllm serve <model> --omni`` in a subprocess, waits for /health, then
exercises /v1/images/generations and /v1/images/edits. First startup on a 24 GB
card takes several minutes because of CPU offload.
"""

from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
from PIL import Image, ImageDraw

pytestmark = pytest.mark.server

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTUP_TIMEOUT = int(os.environ.get("LAB_SERVER_TIMEOUT", "1800"))
SERVER_STEPS = 6
SERVER_SIZE = 512


@dataclass
class ServerHandle:
    base_url: str
    model: str


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server() -> ServerHandle:
    from qwen_lab.runtime import resolve_model

    model = os.environ.get("LAB_MODEL") or resolve_model()
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    vllm = Path(sys.executable).with_name("vllm")
    log_dir = REPO_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "server.log", "w")
    cmd = [
        str(vllm),
        "serve",
        model,
        "--omni",
        "--port",
        str(port),
        "--diffusion-offload-config",
        '{"mode":"layer","components":["dit"]}',
        "--vae-use-tiling",
    ]
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + STARTUP_TIMEOUT
        last_error: Exception | None = None
        healthy = False
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited early (code {process.returncode}); see outputs/server.log")
            try:
                if requests.get(f"{base_url}/health", timeout=3).status_code == 200:
                    healthy = True
                    break
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(5)
        if not healthy:
            raise TimeoutError(f"server not healthy within {STARTUP_TIMEOUT}s ({last_error})")
        yield ServerHandle(base_url=base_url, model=model)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def decode_image(response: requests.Response) -> Image.Image:
    assert response.status_code == 200, response.text
    payload = response.json()
    encoded = payload["data"][0]["b64_json"]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_health(server: ServerHandle) -> None:
    assert requests.get(f"{server.base_url}/health", timeout=5).status_code == 200


def test_text_to_image_endpoint(server: ServerHandle, tmp_path: Path) -> None:
    response = requests.post(
        f"{server.base_url}/v1/images/generations",
        json={
            "model": server.model,
            "prompt": "A red apple on a plain white table",
            "size": f"{SERVER_SIZE}x{SERVER_SIZE}",
            "num_inference_steps": SERVER_STEPS,
            "true_cfg_scale": 1.0,
            "seed": 3,
        },
        timeout=900,
    )
    image = decode_image(response)
    assert image.size == (SERVER_SIZE, SERVER_SIZE)
    assert len(set(image.convert("L").getdata())) > 8, "generated image is (near) uniform"
    image.save(tmp_path / "server-t2i.png")


def test_image_edit_endpoint(server: ServerHandle, tmp_path: Path) -> None:
    reference = Image.new("RGB", (SERVER_SIZE, SERVER_SIZE), (250, 250, 250))
    draw = ImageDraw.Draw(reference)
    draw.ellipse([160, 160, 352, 352], fill=(20, 140, 60))
    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")

    response = requests.post(
        f"{server.base_url}/v1/images/edits",
        data={
            "model": server.model,
            "prompt": "Turn the green circle into a bright yellow star",
            "size": f"{SERVER_SIZE}x{SERVER_SIZE}",
            "num_inference_steps": SERVER_STEPS,
            "true_cfg_scale": 1.0,
            "seed": 5,
        },
        files={"image": ("reference.png", buffer.getvalue(), "image/png")},
        timeout=900,
    )
    image = decode_image(response)
    assert image.size == (SERVER_SIZE, SERVER_SIZE)
    image.save(tmp_path / "server-edit.png")
