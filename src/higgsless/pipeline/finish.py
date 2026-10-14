"""Upscale, interpolate back to the source rate, and colour-match a shot."""

from __future__ import annotations

from higgsless.core.timecode import format_rate
from higgsless.engines.registry import registry
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import ConfigError
from higgsless.pipeline.conform import delivery_size
from higgsless.pipeline.stage import Stage


def _frames_of(stage: Stage, shot_id: str) -> tuple[dict, str, str]:
    render = stage.handle.state.read(f"renders/{shot_id}") or {}
    picked = render.get("picked")
    take = (render.get("takes") or {}).get(picked or "")
    if not take and render.get("mode") != "passthrough":
        raise ConfigError(f"{shot_id} is not rendered", hint=f"higgsless render {shot_id}")
    lipsync = stage.handle.state.read(f"lipsync/{shot_id}") or {}
    frames_list = lipsync.get("frames_list") or (take or {}).get("frames_list")
    return render, frames_list, picked


def finish_shot(stage: Stage, shot_id: str, *, force: bool = False) -> dict:
    render, frames_list, picked = _frames_of(stage, shot_id)
    if frames_list is None:
        state = {"shot": shot_id, "passthrough": True, "take": None}
        stage.handle.state.write(f"finish/{shot_id}", state)
        return state
    policy = stage.frame_policy()
    rate = render.get("rate") or format_rate(policy.gen_rate)
    work = stage.rpath("finish", shot_id)
    current = frames_list
    hold = policy.decimate
    if policy.decimate > 1:
        engine = registry().default_for("interp", stage.project)
        clip = stage.task("frames", {"action": "encode", "list": current, "name": "in.mkv", "rate": rate}, work)
        record = JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, stage.progress).run(
            JobSpec(
                engine,
                {"video": stage.input_rel(clip["outputs"]["clip"][0]), "multiplier": policy.decimate},
                stage.join(work, "interp"),
                f"interp {shot_id}",
            ),
            force=force,
        )
        current = stage.task(
            "frames", {"action": "write_list", "paths": record["outputs"]["frames"], "name": "interp.json"}, work
        )["outputs"]["list"][0]
        hold = 1
    width, height = delivery_size(stage)
    up_engine = registry().default_for("upscale", stage.project)
    clip = stage.task(
        "frames",
        {
            "action": "encode",
            "list": current,
            "name": "up_in.mkv",
            "rate": format_rate(policy.source_rate if hold == 1 else policy.gen_rate),
        },
        work,
    )
    record = JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, stage.progress).run(
        JobSpec(
            up_engine,
            {"video": stage.input_rel(clip["outputs"]["clip"][0]), "width": width, "height": height},
            stage.join(work, "upscale"),
            f"upscale {shot_id}",
        ),
        force=force,
    )
    matched = stage.task(
        "finish",
        {
            "action": "color",
            "frames": record["outputs"]["frames"],
            "reference": stage.join(stage.shot_dir(shot_id), "mezz.mp4"),
            "width": width,
            "height": height,
        },
        work,
    )
    state = {
        "shot": shot_id,
        "frames_list": matched["outputs"]["list"][0],
        "hold": hold,
        "take": picked,
        "width": width,
        "height": height,
    }
    stage.handle.state.write(f"finish/{shot_id}", state)
    return state
