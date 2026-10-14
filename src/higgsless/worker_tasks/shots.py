"""Shot boundary detection: PySceneDetect (adaptive + content + fades), plus TransNetV2 when installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import log, run_task


def scenedetect_cuts(path: str, adaptive: float, content: float, min_len: int) -> tuple[list[int], list[dict]]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector

    video = open_video(path)
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(adaptive_threshold=adaptive, min_scene_len=min_len))
    manager.add_detector(ContentDetector(threshold=content, min_scene_len=min_len))
    manager.detect_scenes(video)
    cuts = sorted({scene[0].get_frames() for scene in manager.get_scene_list()[1:]})

    # Fades to/from black become transitions rather than ordinary cuts.
    fades: list[dict] = []
    video = open_video(path)
    fade_mgr = SceneManager()
    fade_mgr.add_detector(ThresholdDetector(threshold=12, min_scene_len=min_len, fade_bias=0.0))
    fade_mgr.detect_scenes(video)
    for scene in fade_mgr.get_scene_list()[1:]:
        fades.append({"frame": scene[0].get_frames(), "kind": "fade"})
    return cuts, fades


def transnet_cuts(path: str, threshold: float = 0.5) -> tuple[list[int], list[dict]] | None:
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore[import-not-found]
    except ImportError:
        return None
    model = TransNetV2()
    _, single, _ = model.predict_video(path)
    probs = [float(p) for p in single]
    cuts: list[int] = []
    gradual: list[dict] = []
    i = 0
    while i < len(probs):
        if probs[i] > threshold:
            j = i
            while j + 1 < len(probs) and probs[j + 1] > threshold:
                j += 1
            if j - i >= 3:
                gradual.append({"frame": (i + j + 1) // 2, "kind": "dissolve", "length": j - i + 1})
            cuts.append((i + j + 1) // 2 if j > i else i + 1)
            i = j + 1
        else:
            i += 1
    return cuts, gradual


def merge_cuts(cut_lists: list[list[int]], total: int, min_len: int) -> list[int]:
    """Union of cut frames, dropping cuts closer than min_len to a kept one."""
    merged: list[int] = []
    for cut in sorted({c for cuts in cut_lists for c in cuts if 0 < c < total}):
        if merged and cut - merged[-1] < min_len:
            continue
        if total - cut < min_len:
            continue
        merged.append(cut)
    return merged


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    path = params["path"]
    total = int(params["frames"])
    min_len = int(params.get("min_frames", 8))
    cuts, fades = scenedetect_cuts(path, float(params.get("adaptive", 3.0)), float(params.get("content", 27.0)), min_len)
    lists = [cuts]
    transitions = list(fades)
    detectors = ["pyscenedetect"]
    if params.get("transnet", True):
        tn = transnet_cuts(path)
        if tn is not None:
            lists.append(tn[0])
            transitions += tn[1]
            detectors.append("transnetv2")
        else:
            log("TransNetV2 not installed; using PySceneDetect only")
    merged = merge_cuts(lists, total, min_len)
    bounds = [0, *merged, total]
    shots = [{"start": a, "end": b} for a, b in zip(bounds, bounds[1:], strict=False)]
    return {"shots": shots, "transitions": transitions, "detectors": detectors}


if __name__ == "__main__":
    run_task(task)
