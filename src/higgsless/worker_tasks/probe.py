"""ffprobe a video: exact frame rate, frame count, size, audio."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import ffmpeg_bin, run_task, sh


def _rate(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "0"):
        return None
    num, _, den = text.partition("/")
    return Fraction(int(num), int(den or 1))


def probe_file(path: str, count: bool = True) -> dict[str, Any]:
    ffprobe = ffmpeg_bin("ffprobe")
    data = json.loads(sh([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    warnings = []
    r_rate, avg_rate = _rate(video.get("r_frame_rate")), _rate(video.get("avg_frame_rate"))
    rate = r_rate or avg_rate
    if r_rate and avg_rate and abs(float(r_rate - avg_rate)) > 0.01:
        warnings.append(f"variable frame rate suspected (r={r_rate}, avg={avg_rate}); using r_frame_rate")
    frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None
    if frames is None or count:
        out = sh([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                  "stream=nb_read_packets", "-of", "csv=p=0", path])
        frames = int(out.strip().split(",")[0])
    duration = float(data["format"].get("duration") or video.get("duration") or frames / float(rate))
    return {
        "rate": f"{rate.numerator}/{rate.denominator}" if rate.denominator != 1 else str(rate.numerator),
        "frames": frames,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_s": round(duration, 6),
        "has_audio": audio is not None,
        "audio_rate": int(audio["sample_rate"]) if audio else None,
        "codec": video.get("codec_name"),
        "warnings": warnings,
    }


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    info = probe_file(params["path"], count=params.get("count", True))
    if params.get("hash"):
        info["sha256"] = sha256(params["path"])
    return {"probe": info}


if __name__ == "__main__":
    run_task(task)
