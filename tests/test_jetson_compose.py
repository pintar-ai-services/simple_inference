"""Host-independent checks for the Jetson L4T resolver and wheel fallback."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RESOLVER = ROOT / "scripts" / "jetson-compose.sh"
FETCHER = ROOT / "scripts" / "fetch-jetson-wheel.sh"


def resolve(tmp_path: Path, release: str) -> subprocess.CompletedProcess[str]:
    release_file = tmp_path / "nv_tegra_release"
    release_file.write_text(release)
    environment = {**os.environ, "JETSON_RELEASE_FILE": str(release_file)}
    return subprocess.run(
        [str(RESOLVER), "--print-env"], cwd=ROOT, env=environment, text=True, capture_output=True
    )


@pytest.mark.parametrize(
    ("release", "jetpack", "cuda", "base"),
    [
        ("# R36 (release), REVISION: 3.0", "6.0", "12.2", "nvcr.io/nvidia/l4t-jetpack:r36.2.0"),
        ("# R36 (release), REVISION: 4.4", "6.2", "12.6", "nvcr.io/nvidia/l4t-jetpack:r36.4.0"),
        ("# R36 (release), REVISION: 5.2", "6.2", "12.6", "nvcr.io/nvidia/l4t-jetpack:r36.4.0"),
        ("# R39 (release), REVISION: 2.1", "7.2", "13.2", "nvcr.io/nvidia/base/ubuntu:24.04"),
    ],
)
def test_resolver_selects_supported_launcher_wheels(
    tmp_path: Path, release: str, jetpack: str, cuda: str, base: str
):
    result = resolve(tmp_path, release)
    assert result.returncode == 0, result.stderr
    assert f"JETPACK_VERSION={jetpack}" in result.stdout
    assert f"CUDA_VERSION={cuda}" in result.stdout
    assert f"L4T_BASE={base}" in result.stdout


@pytest.mark.parametrize(
    ("release", "message"),
    [
        ("# R36 (release), REVISION: 4.2", "JetPack 6.1"),
        ("# R35 (release), REVISION: 4.1", "JetPack 5"),
        ("not an NVIDIA release file", "could not parse"),
        ("# R40 (release), REVISION: 1.0", "unsupported L4T"),
    ],
)
def test_resolver_rejects_unsupported_releases(tmp_path: Path, release: str, message: str):
    result = resolve(tmp_path, release)
    assert result.returncode != 0
    assert message in result.stderr


def test_wheel_fetcher_uses_fallback_without_network(tmp_path: Path):
    calls = tmp_path / "calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" >> \"$CALLS\"\n"
        "case \"$*\" in *primary*) exit 1;; esac\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = --output ]; then printf wheel > \"$2\"; break; fi\n"
        "  shift\n"
        "done\n"
    )
    fake_curl.chmod(0o755)
    output = tmp_path / "wheel.whl"
    result = subprocess.run(
        [str(FETCHER), str(output), "https://primary/wheel https://fallback/wheel"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "CALLS": str(calls)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "wheel"
    assert "https://primary/wheel" in calls.read_text()
    assert "https://fallback/wheel" in calls.read_text()


def test_jetson_compose_limits_gpu_to_compute_workers():
    compose = (ROOT / "compose.jetson.yaml").read_text()
    # Jetson (including JP7) receives GPU access through runtime: nvidia; the
    # two compute workers explicitly opt in, unlike the API/import stack.
    assert compose.count("runtime: nvidia") == 2
    assert compose.count("NVIDIA_VISIBLE_DEVICES: all") == 2


def test_dockerfile_runs_wheel_helper_without_requiring_host_execute_bits():
    dockerfile = (ROOT / "Dockerfile.jetson").read_text()
    assert "sh /usr/local/bin/fetch-jetson-wheel /tmp/torch.whl" in dockerfile
    assert "sh /usr/local/bin/fetch-jetson-wheel /tmp/torchvision.whl" in dockerfile
