"""serve.sh must bind loopback by default and never echo EXTRA_ARGS."""

import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve.sh"


def _run(tmp_path: Path, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(SCRIPT, tmp_path / "scripts" / "serve.sh")
    stub = tmp_path / ".venv" / "bin" / "vllm"
    stub.write_text('#!/usr/bin/env bash\necho "ARGV: $*"\n')
    stub.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "MODEL": "dummy", **env_extra}
    return subprocess.run(
        ["bash", str(tmp_path / "scripts" / "serve.sh")], env=env, capture_output=True, text=True, check=True
    )


def test_default_binds_loopback(tmp_path):
    out = _run(tmp_path, {}).stdout
    assert "--host 127.0.0.1" in out


def test_host_override_and_no_secret_echo(tmp_path):
    out = _run(tmp_path, {"HOST": "0.0.0.0", "EXTRA_ARGS": "--api-key sk-DUMMY"}).stdout
    assert "--host 0.0.0.0" in out
    banner = out.split("ARGV:")[0]
    assert "sk-DUMMY" not in banner
