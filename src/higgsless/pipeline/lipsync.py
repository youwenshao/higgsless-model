"""Lip-sync the picked take to the new vocal, one shot at a time."""

from __future__ import annotations

from higgsless.core.timecode import format_rate, frame_to_sample, parse_rate
from higgsless.engines.registry import registry
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import ConfigError
from higgsless.pipeline.images import shot_decision
from higgsless.pipeline.stage import Stage


def _slice(stage: Stage, shot_id: str, vocal: str) -> str:
    shot = stage.project.shot(shot_id)
    sample_rate = int(
        stage.ex.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                vocal,
            ]
        ).stdout.strip()
    )
    rate = parse_rate(stage.require_source().rate)
    start = frame_to_sample(shot.start, rate, sample_rate)
    end = frame_to_sample(shot.end, rate, sample_rate)
    dst = stage.rpath("audio", "slices", f"{shot_id}.wav")
    stage.ex.makedirs(stage.rpath("audio", "slices"))
    stage.ex.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            vocal,
            "-ss",
            f"{start / sample_rate:.6f}",
            "-t",
            f"{(end - start) / sample_rate:.6f}",
            "-c:a",
            "pcm_s16le",
            dst,
        ]
    )
    return dst


def lipsync_shot(stage: Stage, shot_id: str, *, force: bool = False) -> dict:
    decision = shot_decision(stage, shot_id)
    render = stage.handle.state.read(f"renders/{shot_id}") or {}
    if not decision.lipsync:
        return {"shot": shot_id, "skipped": "lip-sync off"}
    picked = render.get("picked")
    take = (render.get("takes") or {}).get(picked or "")
    if not take:
        raise ConfigError(f"{shot_id} has no rendered take", hint=f"higgsless render {shot_id}")
    audio_state = stage.handle.state.read("audio") or {}
    vocal = audio_state.get("lead_new")
    if not vocal:
        return {"shot": shot_id, "skipped": "no new vocal"}
    rate = render.get("rate") or format_rate(stage.frame_policy().gen_rate)
    clip = stage.task(
        "frames",
        {"action": "encode", "list": take["frames_list"], "name": "in.mkv", "rate": rate},
        stage.rpath("lipsync", shot_id),
    )["outputs"]["clip"][0]
    audio = _slice(stage, shot_id, vocal)
    engine = registry().get(decision.lipsync_engine or "lipsync.latentsync16")
    params = engine.params_for(
        stage.project.profile.value,
        {
            "video": stage.input_rel(clip),
            "audio": stage.input_rel(audio),
            "width": render.get("width"),
            "height": render.get("height"),
            "length": render.get("frames"),
            "prompt": "a person singing to camera",
            "start_image": stage.input_rel(stage.join(stage.shot_dir(shot_id), "first.png")),
        },
    )
    record = JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, stage.progress).run(
        JobSpec(
            engine,
            {k: v for k, v in params.items() if v is not None},
            stage.rpath("lipsync", shot_id, "out"),
            f"lipsync {shot_id}",
        ),
        force=force,
    )
    frames = record["outputs"]["frames"]
    expected = int(render["frames"])
    if len(frames) != expected:
        fixed = stage.task(
            "frames",
            {"action": "resample", "paths": frames, "count": expected, "rate": rate, "name": "resampled.json"},
            stage.rpath("lipsync", shot_id),
        )
        frames_list = fixed["outputs"]["list"][0]
    else:
        frames_list = stage.task(
            "frames", {"action": "write_list", "paths": frames, "name": "frames.json"}, stage.rpath("lipsync", shot_id)
        )["outputs"]["list"][0]
    state = {"shot": shot_id, "engine": engine.id, "frames_list": frames_list, "take": picked}
    stage.handle.state.write(f"lipsync/{shot_id}", state)
    return state
