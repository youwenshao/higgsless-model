"""Conform: put every shot back on the source timeline at the delivery size, with the chosen audio."""

from __future__ import annotations

from typing import Any

from higgsless.core.timecode import format_rate
from higgsless.core.windows import gen_size
from higgsless.errors import ConfigError
from higgsless.pipeline.stage import Stage

DELIVERY_SHORT = {"720p": 720, "1080p": 1080, "2160p": 2160}


def delivery_size(stage: Stage, resolution: str | None = None) -> tuple[int, int]:
    src = stage.require_source()
    res = resolution or stage.project.delivery.resolution
    if res == "source":
        return src.width - src.width % 2, src.height - src.height % 2
    return gen_size(src.width, src.height, DELIVERY_SHORT[res], 2)


def segments(stage: Stage, *, prefer_finished: bool = True) -> list[dict[str, Any]]:
    project = stage.project
    policy = stage.frame_policy()
    out = []
    for shot in project.shots:
        render = stage.handle.state.read(f"renders/{shot.id}") or {}
        finished = stage.handle.state.read(f"finish/{shot.id}") or {}
        seg: dict[str, Any] = {"id": shot.id, "start": shot.start, "frames": shot.length, "transition": shot.transition}
        picked = render.get("picked")
        lipsynced = (stage.handle.state.read(f"lipsync/{shot.id}") or {}).get("frames_list")
        if prefer_finished and finished.get("frames_list") and finished.get("take") == picked:
            seg.update(
                kind="frames",
                list=finished["frames_list"],
                hold=finished.get("hold", 1),
                engine=render.get("engine"),
                take=picked,
            )
        elif picked and render.get("takes", {}).get(picked):
            seg.update(
                kind="frames",
                list=lipsynced or render["takes"][picked]["frames_list"],
                hold=policy.decimate,
                engine=render.get("engine"),
                take=picked,
            )
        else:
            seg.update(kind="source", path=stage.join(stage.shot_dir(shot.id), "mezz.mp4"), engine="passthrough")
        out.append(seg)
    return out


def conform(
    stage: Stage, *, audio: str = "auto", resolution: str | None = None, pull: bool = False, name: str = "final.mp4"
) -> dict[str, Any]:
    project = stage.project
    if not project.shots:
        raise ConfigError("no shots", hint="higgsless shots detect")
    src = stage.require_source()
    audio_path = None
    if audio in ("auto", "mix"):
        mix = (stage.handle.state.read("audio") or {}).get("final_mix")
        if mix:
            audio_path = mix
        elif audio == "mix":
            raise ConfigError("no final mix yet", hint="higgsless audio mix")
    if audio_path is None and audio in ("auto", "original") and src.has_audio:
        audio_path = (stage.handle.state.read("ingest") or {}).get("audio")
    width, height = delivery_size(stage, resolution)
    segs = segments(stage)
    stage.progress(f"conforming {len(segs)} shots at {width}x{height}")
    res = stage.task(
        "conform",
        {
            "segments": segs,
            "rate": src.rate,
            "width": width,
            "height": height,
            "audio": audio_path,
            "codec": project.delivery.codec,
            "crf": project.delivery.crf,
            "name": name,
        },
        stage.rpath("conform"),
    )
    result = {
        "video": res["outputs"]["video"][0],
        "otio": res["outputs"]["otio"][0],
        "frames": res["frames"],
        "duration_s": res["duration_s"],
        "width": width,
        "height": height,
        "audio": audio_path,
        "rate": format_rate(stage.frame_policy().source_rate),
        "shots": {s["id"]: s.get("engine") for s in segs},
    }
    if pull:
        result["local_video"] = str(stage.pull(result["video"], name))
        result["local_otio"] = str(stage.pull(result["otio"], "timeline.otio"))
    stage.handle.state.write("conform", result)
    return result
