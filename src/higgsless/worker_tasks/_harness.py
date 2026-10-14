"""Shared entry point for worker tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

TaskFn = Callable[[dict[str, Any], Path], dict[str, Any]]


def run_task(fn: TaskFn) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--params-file", help="JSON file with the params (for large payloads)")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = Path(args.params_file).read_text(encoding="utf-8") if args.params_file else args.params
        result = fn(json.loads(raw), out_dir) or {}
        result.setdefault("ok", True)
    except Exception as exc:  # noqa: BLE001 - reported through the result file
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    sys.exit(0)


def log(message: str) -> None:
    print(message, flush=True)


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found on PATH")
    return found


def sh(argv: list[str], *, capture: bool = True) -> str:
    proc = subprocess.run(argv, capture_output=capture, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)[:300]}\n{tail}")
    return proc.stdout or ""
