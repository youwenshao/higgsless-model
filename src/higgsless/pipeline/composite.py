"""Composite mode (experimental): Wan-Animate-2 render matted over a VACE clean plate of the original set."""

from __future__ import annotations

from typing import Any

from higgsless.core.timecode import format_rate
from higgsless.core.windows import plan_windows
from higgsless.engines.registry import registry
from higgsless.engines.router import Decision
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import JobFailed
from higgsless.pipeline.analyze import track_clip
from higgsless.pipeline.stage import Stage


def render_composite(
    stage: Stage, shot_id: str, decision: Decision, *, takes: int | None, force: bool, gate: bool
) -> dict[str, Any]:
    from higgsless.pipeline.render import gen_clip, gen_dims, render_with, window_clip

    project = stage.project
    runner = JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, progress=stage.progress)
    plate_engine, _ = registry().resolve("plate", project)
    width, height = gen_dims(stage, plate_engine)
    clip, total = gen_clip(stage, shot_id, width, height)
    rate = format_rate(stage.frame_policy().gen_rate)
    work = stage.join(stage.shot_dir(shot_id), "composite")

    # 1. Mask of the cast performers (from the analysis tracks), then a clean plate without them.
    analysis = stage.handle.state.read(f"analysis/{shot_id}") or {}
    colors = [p["color"] for p in analysis.get("people", []) if p["track"] in decision.cast]
    mask = stage.task(
        "composite",
        {
            "action": "mask_video",
            "mask_dir": analysis.get("mask_dir"),
            "colors": colors,
            "width": width,
            "height": height,
            "rate": rate,
            "grow": 16,
            "name": "performer_mask.mkv",
        },
        work,
    )
    mask_clip = mask["outputs"]["clip"][0]
    plate_frames: list[str] = []
    for w in plan_windows(total, window=plate_engine.constraints.window or 81, overlap=0):
        win = window_clip(stage, shot_id, clip, total, w, width, height)
        win_mask = window_clip(stage, shot_id, mask_clip, total, w, width, height)
        params = plate_engine.params_for(
            project.profile.value,
            {
                "source_video": stage.input_rel(win),
                "mask_video": stage.input_rel(win_mask),
                "width": width,
                "height": height,
                "length": w.length,
            },
        )
        record = runner.run(
            JobSpec(
                plate_engine,
                params,
                stage.rpath("renders", shot_id, "plate", f"w{w.index:02d}"),
                f"plate {shot_id} w{w.index}",
            ),
            force=force,
        )
        plate_frames += record["outputs"]["frames"][: w.emit_count]
    if len(plate_frames) != total:
        raise JobFailed(f"{shot_id}: clean plate has {len(plate_frames)} frames, expected {total}")
    plate_list = stage.task(
        "frames",
        {"action": "write_list", "paths": plate_frames, "name": "frames.json"},
        stage.rpath("renders", shot_id, "plate"),
    )["outputs"]["list"][0]

    # 2. The character render on the keyframe, 3. its SAM3 matte, 4. the composite over the plate.
    rebuild = Decision(shot=shot_id, mode="rebuild", engine="rebuild.wan_animate2", cast=decision.cast)
    state = render_with(stage, shot_id, rebuild, takes=takes, force=force, gate=gate)
    for take, info in state["takes"].items():
        take_dir = stage.rpath("renders", shot_id, "composite", f"take{take}")
        render_clip = stage.task(
            "frames", {"action": "encode", "list": info["frames_list"], "name": "render.mkv", "rate": rate}, take_dir
        )["outputs"]["clip"][0]
        matte_dir = track_clip(
            stage, render_clip, stage.join(take_dir, "matte"), f"matte {shot_id} take{take}", force=force
        )
        comp = stage.task(
            "composite",
            {
                "action": "composite",
                "render_list": info["frames_list"],
                "plate_list": plate_list,
                "matte_dir": matte_dir,
                "width": state["width"],
                "height": state["height"],
            },
            take_dir,
        )
        info["frames_list"] = comp["outputs"]["list"][0]
        info["engine"] = "composite.animate2_matte"
    state["mode"] = "composite"
    stage.handle.state.write(f"renders/{shot_id}", state)
    return state
