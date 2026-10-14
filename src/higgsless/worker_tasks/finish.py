"""Colour-match rendered frames to the source shot (per-channel mean and spread)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from higgsless.worker_tasks._harness import ffmpeg_bin, run_task, sh


def _stats(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = frame.reshape(-1, frame.shape[-1]).astype(np.float32)
    return flat.mean(0), flat.std(0) + 1e-6


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    frames = params["frames"]
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "%06d.png")
        sh([ffmpeg_bin(), "-y", "-v", "error", "-i", params["reference"], pattern])
        refs = sorted(Path(tmp).glob("*.png"))
    size = (int(params["width"]), int(params["height"]))
    dst_dir = out / "frames"
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, path in enumerate(frames):
        image = np.asarray(Image.open(path).convert("RGB").resize(size)).astype(np.float32)
        ref = np.asarray(Image.open(refs[min(i, len(refs) - 1)]).convert("RGB").resize(size)).astype(np.float32)
        mean, std = _stats(image)
        ref_mean, ref_std = _stats(ref)
        matched = np.clip((image - mean) / std * ref_std + ref_mean, 0, 255).astype(np.uint8)
        dst = dst_dir / f"frame_{i:06d}.png"
        Image.fromarray(matched).save(dst)
        written.append(str(dst))
    listing = out / "frames.json"
    listing.write_text(json.dumps(written), encoding="utf-8")
    return {"count": len(written), "outputs": {"list": [str(listing)]}}


if __name__ == "__main__":
    run_task(task)
