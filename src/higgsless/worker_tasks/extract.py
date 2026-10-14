"""Extract derived media from the source: audio, proxy, per-shot mezzanines, generator clips, frames."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from higgsless.core.timecode import parse_rate
from higgsless.video import ff
from higgsless.worker_tasks._harness import log, run_task, sh
from higgsless.worker_tasks.probe import probe_file


def _run(argv: list[str]) -> None:
    sh(argv)


def _check_frames(path: str, expected: int) -> None:
    got = probe_file(path)["frames"]
    if got != expected:
        raise RuntimeError(f"{path}: expected {expected} frames, got {got}")


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    action = params["action"]
    src = params["source"]
    outputs: dict[str, list[str]] = {}

    if action == "ingest":
        audio = out / "original.wav"
        proxy = out / "proxy.mp4"
        jobs = [ff.proxy(src, str(proxy), int(params.get("proxy_short_edge", 480)))]
        if params.get("has_audio", True):
            jobs.append(ff.audio_wav(src, str(audio)))
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(_run, jobs))
        _check_frames(str(proxy), int(params["frames"]))
        outputs = {"proxy": [str(proxy)]}
        if params.get("has_audio", True):
            outputs["audio"] = [str(audio)]

    elif action == "shots":
        # Per-shot mezzanine (near-lossless, source resolution) plus the first frame as PNG.
        rate = parse_rate(params["rate"])
        shots = params["shots"]  # [{"id", "start", "end"}]

        def one(shot: dict[str, Any]) -> tuple[str, str]:
            shot_dir = out / shot["id"]
            shot_dir.mkdir(parents=True, exist_ok=True)
            mezz = shot_dir / "mezz.mp4"
            count = shot["end"] - shot["start"]
            if not (mezz.exists() and probe_file(str(mezz))["frames"] == count):
                sh(ff.cut_clip(src, str(mezz), shot["start"], count, rate))
                _check_frames(str(mezz), count)
            first = shot_dir / "first.png"
            sh(ff.frame_png(str(mezz), str(first), 0, rate))
            log(f"{shot['id']}: {count} frames")
            return str(mezz), str(first)

        with ThreadPoolExecutor(int(params.get("workers", 4))) as pool:
            results = list(pool.map(one, shots))
        outputs = {"mezz": [r[0] for r in results], "first": [r[1] for r in results]}

    elif action == "gen_clip":
        # Generator input for one shot: decimated, scaled, optionally trimmed to one window (lossless).
        dst = out / params["name"]
        vf = ff.gen_filter(int(params["width"]), int(params["height"]), int(params.get("decimate", 1)),
                           params.get("start"), params.get("end"))
        sh(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf, "-fps_mode", "passthrough", "-an",
            *ff.LOSSLESS_ARGS, str(dst)])
        if params.get("expect_frames"):
            _check_frames(str(dst), int(params["expect_frames"]))
        outputs = {"clip": [str(dst)]}

    elif action == "frames":
        rate = parse_rate(params["rate"])
        paths = []
        for frame in params["frames"]:
            dst = out / f"frame_{int(frame):06d}.png"
            sh(ff.frame_png(src, str(dst), int(frame), rate))
            paths.append(str(dst))
        outputs = {"frames": paths}
    else:
        raise ValueError(f"unknown action {action!r}")
    return {"outputs": outputs}


if __name__ == "__main__":
    run_task(task)
