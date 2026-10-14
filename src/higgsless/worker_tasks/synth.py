"""Generate synthetic test media (for `bench` and smoke tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import ffmpeg_bin, run_task, sh


def synth(params: dict[str, Any], out: Path) -> dict[str, Any]:
    width = int(params.get("width", 832))
    height = int(params.get("height", 480))
    frames = int(params.get("frames", 81))
    rate = str(params.get("rate", "24"))
    ffmpeg = ffmpeg_bin()
    video = out / "driving.mp4"
    sh(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={rate}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(video),
        ]
    )
    image = out / "reference.png"
    sh(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x8899aa:s={min(width, height)}x{max(width, height)}",
            "-frames:v",
            "1",
            str(image),
        ]
    )
    return {"outputs": {"driving_video": [str(video)], "reference_image": [str(image)]}}


if __name__ == "__main__":
    run_task(synth)
