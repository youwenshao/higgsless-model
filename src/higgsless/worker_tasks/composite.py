"""Masks from SAM3 colored tracks, and alpha compositing a render over a clean plate."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from higgsless.worker_tasks._harness import run_task
from higgsless.worker_tasks.frames import encode


def binary_mask(path: Path, colors: list[list[int]] | None, size: tuple[int, int], grow: int) -> Image.Image:
    frame = np.asarray(Image.open(path).convert("RGB"))
    quant = (frame // 16) * 16
    if colors:
        mask = np.zeros(frame.shape[:2], dtype=bool)
        for color in colors:
            mask |= np.all(quant == np.array(color), axis=-1)
    else:
        mask = frame.max(axis=-1) > 24
    image = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    if grow > 0:
        image = image.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    return image


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    action = params["action"]
    size = (int(params["width"]), int(params["height"]))
    if action == "mask_video":
        masks = sorted(Path(params["mask_dir"]).glob("*.png"))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, path in enumerate(masks):
                dst = Path(tmp) / f"{i:06d}.png"
                binary_mask(path, params.get("colors"), size, int(params.get("grow", 12))).convert("RGB").save(dst)
                paths.append(str(dst))
            name = params.get("name", "mask.mkv")
            encode(paths, str(out / name), params["rate"])
        return {"outputs": {"clip": [str(out / name)]}}

    if action == "composite":
        render = json.loads(Path(params["render_list"]).read_text(encoding="utf-8"))
        plate = json.loads(Path(params["plate_list"]).read_text(encoding="utf-8"))
        masks = sorted(Path(params["matte_dir"]).glob("*.png"))
        n = min(len(render), len(plate), len(masks))
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        feather = int(params.get("feather", 4))
        paths = []
        for i in range(n):
            fg = Image.open(render[i]).convert("RGB").resize(size)
            bg = Image.open(plate[i]).convert("RGB").resize(size)
            alpha = binary_mask(masks[i], None, size, int(params.get("grow", 2)))
            if feather:
                alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
            frame = Image.composite(fg, bg, alpha)
            dst = frames_dir / f"frame_{i:06d}.png"
            frame.save(dst)
            paths.append(str(dst))
        listing = out / "frames.json"
        listing.write_text(json.dumps(paths), encoding="utf-8")
        return {"count": n, "outputs": {"list": [str(listing)]}}
    raise ValueError(f"unknown action {action!r}")


if __name__ == "__main__":
    run_task(task)
