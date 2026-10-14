"""Frame-list utilities: encode a list of PNGs into a clip, measure takes, build review sheets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from higgsless.qa.contact_sheet import contact_sheet
from higgsless.worker_tasks._harness import ffmpeg_bin, run_task, sh


def _link_sequence(paths: list[str], folder: Path) -> str:
    for i, path in enumerate(paths, start=1):
        target = folder / f"{i:06d}{Path(path).suffix}"
        try:
            os.symlink(path, target)
        except (OSError, NotImplementedError):
            shutil.copyfile(path, target)
    return str(folder / f"%06d{Path(paths[0]).suffix}")


def encode(paths: list[str], dst: str, rate: str, lossless: bool = True, vf: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pattern = _link_sequence(paths, Path(tmp))
        argv = [ffmpeg_bin(), "-y", "-v", "error", "-framerate", rate, "-i", pattern]
        if vf:
            argv += ["-vf", vf]
        if lossless:
            argv += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"]
        else:
            argv += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
        sh([*argv, "-r", rate, dst])


def _gray(path: str, width: int = 160) -> np.ndarray:
    im = Image.open(path).convert("L")
    h = max(2, round(im.height * width / im.width))
    return np.asarray(im.resize((width, h)), dtype=np.float32) / 255.0


def flicker(paths: list[str]) -> float:
    """Median absolute frame-to-frame change (0-1)."""
    diffs = []
    prev = None
    for path in paths:
        cur = _gray(path)
        if prev is not None and cur.shape == prev.shape:
            diffs.append(float(np.abs(cur - prev).mean()))
        prev = cur
    return float(np.median(diffs)) if diffs else 0.0


def read_video_frames(video: str, out: Path, every: int = 1) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "%06d.png")
    vf = f"select='not(mod(n\\,{every}))'" if every > 1 else None
    argv = [ffmpeg_bin(), "-y", "-v", "error", "-i", video]
    if vf:
        argv += ["-vf", vf, "-fps_mode", "passthrough"]
    sh([*argv, pattern])
    return sorted(str(p) for p in out.glob("*.png"))


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    action = params["action"]
    if action == "encode":
        paths = params.get("paths") or json.loads(Path(params["list"]).read_text(encoding="utf-8"))
        dst = str(out / params["name"])
        encode(paths, dst, params["rate"], lossless=params.get("lossless", True), vf=params.get("vf"))
        return {"outputs": {"clip": [dst]}}

    if action == "write_list":
        dst = out / params["name"]
        dst.write_text(json.dumps(params["paths"]), encoding="utf-8")
        return {"outputs": {"list": [str(dst)]}}

    if action == "resample":
        paths = params["paths"]
        count = int(params["count"])
        if not paths:
            raise RuntimeError("no frames to resample")
        index = [round(i * (len(paths) - 1) / max(1, count - 1)) for i in range(count)]
        chosen = [paths[i] for i in index]
        listing = out / params["name"]
        listing.write_text(json.dumps(chosen), encoding="utf-8")
        return {"outputs": {"list": [str(listing)]}}

    if action == "metrics":
        paths = json.loads(Path(params["list"]).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            source = read_video_frames(params["source_clip"], Path(tmp))
            n = min(len(paths), len(source))
            render_flicker = flicker(paths[:n])
            source_flicker = flicker(source[:n])
            picks = sorted({0, n // 3, (2 * n) // 3, max(0, n - 1)})
            items = []
            for i in picks:
                items.append((source[i], f"source {i}"))
                items.append((paths[i], f"render {i}"))
            sheet = out / params.get("sheet_name", "review.jpg")
            contact_sheet(items, sheet, columns=2, cell_width=360, title=params.get("title"))
            identity = None
            if params.get("keyframe"):
                from higgsless.worker_tasks.embed import embed

                vecs, method = embed([params["keyframe"]] + [paths[i] for i in picks])
                sims = (vecs[1:] @ vecs[0]) / (np.linalg.norm(vecs[1:], axis=1) * np.linalg.norm(vecs[0]) + 1e-8)
                identity = {"method": method, "mean": round(float(sims.mean()), 4), "min": round(float(sims.min()), 4)}
        return {
            "frames": len(paths),
            "expected": params.get("expected"),
            "flicker": round(render_flicker, 5),
            "source_flicker": round(source_flicker, 5),
            "flicker_ratio": round(render_flicker / source_flicker, 3) if source_flicker > 1e-6 else None,
            "identity": identity,
            "outputs": {"sheet": [str(sheet)]},
        }
    raise ValueError(f"unknown action {action!r}")


if __name__ == "__main__":
    run_task(task)
