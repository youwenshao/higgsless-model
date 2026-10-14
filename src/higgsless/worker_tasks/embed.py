"""Image embeddings for cast clustering and identity QA (DINOv2; colour histograms as a fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from higgsless.worker_tasks._harness import log, run_task


def histogram_embedding(image: Image.Image) -> np.ndarray:
    small = np.asarray(image.convert("RGB").resize((64, 128)), dtype=np.float32) / 255.0
    hist = [np.histogram(small[..., c], bins=16, range=(0, 1))[0] for c in range(3)]
    vec = np.concatenate(hist).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def dino_embeddings(images: list[Image.Image], model_id: str) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = processor(images=images[i : i + 16], return_tensors="pt").to(device)
            cls = model(**batch).last_hidden_state[:, 0]
            vecs.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(vecs)


def embed(paths: list[str], model_id: str = "facebook/dinov2-base") -> tuple[np.ndarray, str]:
    images = [Image.open(p).convert("RGB") for p in paths]
    try:
        return dino_embeddings(images, model_id), model_id
    except Exception as exc:  # noqa: BLE001 - fall back when transformers/torch are unavailable
        log(f"DINOv2 unavailable ({type(exc).__name__}); using colour histograms")
        return np.stack([histogram_embedding(im) for im in images]), "histogram"


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    paths = [p for p in params["paths"] if Path(p).exists()]
    if not paths:
        return {"embeddings": [], "paths": [], "method": None}
    vecs, method = embed(paths, params.get("model", "facebook/dinov2-base"))
    return {"embeddings": vecs.round(5).tolist(), "paths": paths, "method": method}


if __name__ == "__main__":
    run_task(task)
