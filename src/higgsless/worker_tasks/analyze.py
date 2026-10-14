"""Per-shot analysis: people (from SAM3 colored identity masks), face sizes, camera motion, person crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from higgsless.worker_tasks._harness import log, run_task


def _palette_ids(frame: np.ndarray, min_pixels: int) -> dict[tuple[int, int, int], np.ndarray]:
    """Map each non-black color in a colored mask frame to its boolean mask."""
    flat = frame.reshape(-1, 3)
    nonblack = flat.max(axis=1) > 24
    colors, counts = np.unique((flat[nonblack] // 16) * 16, axis=0, return_counts=True)
    out = {}
    quant = (frame // 16) * 16
    for color, count in zip(colors, counts, strict=True):
        if count < min_pixels:
            continue
        key = tuple(int(c) for c in color)
        out[key] = np.all(quant == color, axis=-1)
    return out


def people_from_masks(mask_paths: list[Path], min_frac: float = 0.002) -> list[dict[str, Any]]:
    tracks: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx, path in enumerate(mask_paths):
        frame = np.asarray(Image.open(path).convert("RGB"))
        h, w = frame.shape[:2]
        for color, mask in _palette_ids(frame, int(min_frac * h * w)).items():
            ys, xs = np.nonzero(mask)
            area = mask.mean()
            box = [int(xs.min()) / w, int(ys.min()) / h, int(xs.max() + 1) / w, int(ys.max() + 1) / h]
            t = tracks.setdefault(
                color,
                {
                    "color": list(color),
                    "first_frame": idx,
                    "frames": 0,
                    "areas": [],
                    "best_frame": idx,
                    "best_area": 0.0,
                    "best_box": box,
                    "first_box": box,
                },
            )
            t["frames"] += 1
            t["areas"].append(float(area))
            if area > t["best_area"]:
                t.update(best_frame=idx, best_area=float(area), best_box=box)
    n = max(1, len(mask_paths))
    people = []
    for t in sorted(tracks.values(), key=lambda t: (t["first_frame"], t["first_box"][0])):
        people.append(
            {
                "color": t["color"],
                "first_frame": t["first_frame"],
                "presence": round(t["frames"] / n, 3),
                "mean_area": round(float(np.mean(t["areas"])), 4),
                "best_frame": t["best_frame"],
                "best_box": [round(v, 4) for v in t["best_box"]],
                "first_box": [round(v, 4) for v in t["first_box"]],
                "height_frac": round(t["best_box"][3] - t["best_box"][1], 3),
            }
        )
    for i, p in enumerate(people, start=1):
        p["track"] = str(i)
    return people


def _read_frames(video: str, step: int, width: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            frames.append(cv2.resize(frame, (width, max(2, int(h * width / w)))))
        i += 1
    cap.release()
    return frames


def camera_motion(frames: list[np.ndarray]) -> float:
    import cv2

    mags = []
    prev = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.median(np.linalg.norm(flow, axis=-1))) / gray.shape[1])
        prev = gray
    return round(float(np.median(mags)) * 100, 3) if mags else 0.0


def face_sizes(frames: list[np.ndarray]) -> dict[str, Any]:
    try:
        import mediapipe as mp
    except ImportError:
        return {"available": False}
    detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    fracs, counts = [], []
    for frame in frames:
        result = detector.process(frame[:, :, ::-1])
        dets = result.detections or []
        counts.append(len(dets))
        if dets:
            fracs.append(max(d.location_data.relative_bounding_box.height for d in dets))
    detector.close()
    if not fracs:
        return {"available": True, "max_frac": 0.0, "median_frac": 0.0, "max_faces": max(counts, default=0)}
    return {
        "available": True,
        "max_frac": round(float(max(fracs)), 3),
        "median_frac": round(float(np.median(fracs)), 3),
        "max_faces": int(max(counts)),
        "face_presence": round(len(fracs) / max(1, len(frames)), 3),
    }


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    video = params["video"]
    mask_dir = Path(params["mask_dir"]) if params.get("mask_dir") else None
    masks = sorted(mask_dir.glob("*.png")) if mask_dir and mask_dir.exists() else []
    people = people_from_masks(masks) if masks else []
    frames = _read_frames(video, int(params.get("step", 4)), 320)
    result: dict[str, Any] = {"people": people, "motion": camera_motion(frames), "faces": face_sizes(frames)}

    crops = []
    if people and params.get("mezz"):
        import cv2

        cap = cv2.VideoCapture(params["mezz"])
        for p in people:
            cap.set(cv2.CAP_PROP_POS_FRAMES, p["best_frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = p["best_box"]
            crop = frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
            if crop.size == 0:
                continue
            path = out / f"person_{p['track']}.jpg"
            cv2.imwrite(str(path), crop)
            p["crop"] = str(path)
            crops.append(str(path))
        cap.release()
    log(f"{len(people)} people, motion {result['motion']}, faces {result['faces'].get('max_frac')}")
    return {**result, "outputs": {"crops": crops}}


if __name__ == "__main__":
    run_task(task)
