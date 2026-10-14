"""Conform: assemble shots frame-exactly at the source rate, recreate dissolves, mux audio, export OTIO."""

from __future__ import annotations

import json
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import ffmpeg_bin, log, run_task, sh
from higgsless.worker_tasks.frames import encode
from higgsless.worker_tasks.probe import probe_file

INTERMEDIATE = ["-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p"]


def _scale(width: int, height: int) -> str:
    return f"scale={width}:{height}:flags=lanczos,setsar=1"


def _hold(paths: list[str], k: int, count: int) -> list[str]:
    """Repeat each frame k times (restores the source rate after decimation), trimmed to count."""
    if k <= 1:
        return paths[:count]
    held = [p for p in paths for _ in range(k)]
    if len(held) < count:
        held += [held[-1]] * (count - len(held))
    return held[:count]


def build_segment(seg: dict[str, Any], dst: Path, rate: str, width: int, height: int) -> None:
    count = int(seg["frames"])
    if seg["kind"] == "frames":
        paths = json.loads(Path(seg["list"]).read_text(encoding="utf-8"))
        paths = _hold(paths, int(seg.get("hold", 1)), count)
        if len(paths) != count:
            raise RuntimeError(f"{seg['id']}: {len(paths)} frames for a {count}-frame shot")
        encode(paths, str(dst), rate, lossless=False, vf=_scale(width, height))
    else:
        sh(
            [
                ffmpeg_bin(),
                "-y",
                "-v",
                "error",
                "-i",
                seg["path"],
                "-vf",
                _scale(width, height),
                "-frames:v",
                str(count),
                "-an",
                *INTERMEDIATE,
                "-r",
                rate,
                str(dst),
            ]
        )
    got = probe_file(str(dst))["frames"]
    if got != count:
        raise RuntimeError(f"{seg['id']}: segment has {got} frames, expected {count}")


def apply_dissolve(prev: Path, cur: Path, frames: int, rate: str, workdir: Path) -> None:
    """Recreate a dissolve across a cut without changing timing.

    The last half of `prev` blends towards `cur`'s first frame and the first half of `cur` blends from
    `prev`'s last frame (frame-held crossfade), so both segments keep their exact lengths.
    """
    if frames < 2:
        return
    half_a, half_b = frames // 2, frames - frames // 2
    a_frames = probe_file(str(prev))["frames"]
    b_frames = probe_file(str(cur))["frames"]
    half_a, half_b = min(half_a, a_frames - 1), min(half_b, b_frames - 1)
    ff = ffmpeg_bin()
    for src, other, n, at_end in ((prev, cur, half_a, True), (cur, prev, half_b, False)):
        if n <= 0:
            continue
        held = workdir / f"held_{src.stem}.png"
        pick = "select='eq(n\\,0)'" if at_end else f"select='eq(n\\,{probe_file(str(other))['frames'] - 1})'"
        sh([ff, "-y", "-v", "error", "-i", str(other), "-vf", pick, "-frames:v", "1", str(held)])
        total = a_frames if at_end else b_frames
        start = total - n if at_end else 0
        # alpha ramps from 0 to 0.5 into the cut (prev) and 0.5 to 0 out of it (cur)
        expr = f"if(between(N,{start},{start + n - 1}),"
        expr += f"(N-{start}+1)/({n}*2)" if at_end else f"0.5-(N)/({n}*2)"
        expr += ",0)"
        out = workdir / f"dissolved_{src.name}"
        graph = (
            f"[1:v]scale=iw:ih,format=yuva420p,colorchannelmixer=aa=1[h];"
            f"[0:v][h]blend=all_expr='A*(1-({expr}))+B*({expr})':shortest=0[v]"
        )
        sh(
            [
                ff,
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-loop",
                "1",
                "-i",
                str(held),
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-frames:v",
                str(total),
                *INTERMEDIATE,
                "-r",
                rate,
                str(out),
            ]
        )
        out.replace(src)


def write_otio(segments: list[dict[str, Any]], rate: Fraction, dst: Path, media: str) -> None:
    import opentimelineio as otio

    timeline = otio.schema.Timeline(name="higgsless conform")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)
    rate_f = float(rate)
    for seg in segments:
        ref = otio.schema.ExternalReference(target_url=media)
        start = otio.opentime.RationalTime(seg["start"], rate_f)
        duration = otio.opentime.RationalTime(seg["frames"], rate_f)
        clip = otio.schema.Clip(
            name=f"{seg['id']} ({seg.get('engine') or seg['kind']})",
            media_reference=ref,
            source_range=otio.opentime.TimeRange(start, duration),
        )
        clip.metadata["higgsless"] = {k: seg.get(k) for k in ("id", "kind", "engine", "take", "transition")}
        track.append(clip)
    otio.adapters.write_to_file(timeline, str(dst))


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    rate = params["rate"]
    num, _, den = rate.partition("/")
    rate_fr = Fraction(int(num), int(den or 1))
    width, height = int(params["width"]), int(params["height"])
    segments = params["segments"]
    seg_dir = out / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, seg in enumerate(segments):
        dst = seg_dir / f"{i:04d}_{seg['id']}.mp4"
        build_segment(seg, dst, rate, width, height)
        log(f"{seg['id']}: {seg['frames']} frames ({seg['kind']})")
        files.append(dst)
    with tempfile.TemporaryDirectory() as tmp:
        for i, seg in enumerate(segments):
            transition = seg.get("transition") or "cut"
            if i > 0 and transition.startswith("dissolve:"):
                apply_dissolve(files[i - 1], files[i], int(transition.split(":")[1]), rate, Path(tmp))
    listing = out / "segments.txt"
    listing.write_text("".join(f"file '{f.as_posix()}'\n" for f in files), encoding="utf-8")
    timeline = out / "timeline.mp4"
    sh(
        [
            ffmpeg_bin(),
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(timeline),
        ]
    )
    total = sum(int(s["frames"]) for s in segments)
    got = probe_file(str(timeline))["frames"]
    if got != total:
        raise RuntimeError(f"timeline has {got} frames, expected {total}")
    duration = float(Fraction(total) / rate_fr)
    final = out / params.get("name", "final.mp4")
    vcodec = {"h264": ["-c:v", "libx264"], "hevc": ["-c:v", "libx265", "-tag:v", "hvc1"]}[params.get("codec", "h264")]
    argv = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(timeline)]
    if params.get("audio"):
        argv += ["-i", params["audio"], "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "320k", "-af", "apad"]
    argv += [
        *vcodec,
        "-crf",
        str(params.get("crf", 18)),
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-r",
        rate,
        "-t",
        f"{duration:.6f}",
        str(final),
    ]
    sh(argv)
    otio_path = out / "timeline.otio"
    write_otio(segments, rate_fr, otio_path, final.as_posix())
    return {
        "frames": total,
        "duration_s": round(duration, 6),
        "outputs": {"video": [str(final)], "otio": [str(otio_path)], "timeline": [str(timeline)]},
    }


if __name__ == "__main__":
    run_task(task)
