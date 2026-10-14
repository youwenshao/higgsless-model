from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def projects_home(tmp_path, monkeypatch):
    """Isolated projects folder."""
    home = tmp_path / "projects"
    home.mkdir()
    monkeypatch.setenv("HIGGSLESS_PROJECTS", str(home))
    return home


@pytest.fixture()
def local_config(tmp_path, monkeypatch):
    """A workstation config that runs everything locally (worker tasks natively)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    cfg = tmp_path / "workstation.yaml"
    cfg.write_text(
        "\n".join(
            [
                "executor: local",
                "comfy: {url: 'http://127.0.0.1:1'}",
                f"data_root: '{data_root.as_posix()}'",
                f"worker_dir: '{REPO.as_posix()}'",
                f"hf_cache: '{(tmp_path / 'hf').as_posix()}'",
                "tools_mode: native",
                f"native_python: '{Path(sys.executable).as_posix()}'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HIGGSLESS_CONFIG", str(cfg))
    return cfg


def has_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
