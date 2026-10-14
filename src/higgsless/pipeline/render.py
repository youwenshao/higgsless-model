"""Render shots: generator clips, windows chained per engine, takes, QA, picks."""

from __future__ import annotations

from typing import Any

from higgsless.core.cache import hash_obj
from higgsless.core.manifest import Mode
from higgsless.core.timecode import format_rate
from higgsless.core.windows import Window, gen_size, length_4n1, plan_windows, resolve_short_edge
from higgsless.engines.registry import registry
from higgsless.engines.router import Decision
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.engines.spec import EngineSpec
from higgsless.errors import ConfigError, JobFailed
from higgsless.pipeline.images import require_approved, shot_decision
from higgsless.pipeline.stage import Stage

NEEDS_KEYFRAME = {Mode.SWAP.value, Mode.REBUILD.value, Mode.COMPOSITE.value}


def _runner(stage: Stage) -> JobRunner:
    return JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, progress=stage.progress)


def gen_dims(stage: Stage, engine: EngineSpec) -> tuple[int, int]:
    src = stage.require_source()
    short = engine.constraints.preferred_height or resolve_short_edge(stage.project.resolution)
    if engine.constraints.preferred_height and stage.project.profile.value == "draft":
        short = resolve_short_edge("480p")
    return gen_size(src.width, src.height, short, engine.constraints.dim_multiple)


def gen_clip(stage: Stage, shot_id: str, width: int, height: int) -> tuple[str, int]:
    """Whole shot at generator size and rate (lossless). Returns (path, frame count)."""
    policy = stage.frame_policy()
    shot = stage.project.shot(shot_id)
    frames = policy.gen_frames(shot.length)
    name = f"gen_{width}x{height}_k{policy.decimate}.mkv"
    path = stage.join(stage.shot_dir(shot_id), name)
    if not stage.ex.exists(path):
        stage.task(
            "extract",
            {
                "action": "gen_clip",
                "source": stage.join(stage.shot_dir(shot_id), "mezz.mp4"),
                "name": name,
                "width": width,
                "height": height,
                "decimate": policy.decimate,
                "expect_frames": frames,
            },
            stage.shot_dir(shot_id),
        )
    return path, frames


def window_clip(stage: Stage, shot_id: str, clip: str, total: int, w: Window, width: int, height: int) -> str:
    end = min(total, w.start + w.length)
    name = f"win_{width}x{height}_{w.start}_{end}.mkv"
    path = stage.join(stage.shot_dir(shot_id), "windows", name)
    if not stage.ex.exists(path):
        stage.task(
            "extract",
            {
                "action": "gen_clip",
                "source": clip,
                "name": name,
                "width": width,
                "height": height,
                "start": w.start,
                "end": end,
                "expect_frames": end - w.start,
            },
            stage.join(stage.shot_dir(shot_id), "windows"),
        )
    return path


def _character_sheet(stage: Stage, decision: Decision, index: int = 0) -> str | None:
    names = list(dict.fromkeys(decision.cast.values()))
    if len(names) <= index:
        return None
    sheet = stage.project.characters[names[index]].sheet
    return stage.input_rel(stage.rpath(sheet)) if sheet else None


def engine_params(
    stage: Stage,
    engine: EngineSpec,
    decision: Decision,
    keyframe: dict[str, Any] | None,
    w: Window,
    clip_rel: str,
    prev_rel: str | None,
    width: int,
    height: int,
    seed: int,
) -> dict:
    """Map the shot/window onto an engine's logical parameters."""
    project = stage.project
    shot = project.shot(decision.shot)
    analysis = stage.handle.state.read(f"analysis/{shot.id}") or {}
    prompt = (keyframe or {}).get("caption") or (shot.prompt or "a person performing")
    common = {"width": width, "height": height, "length": w.length, "seed": seed}
    key = engine.id
    if key == "rebuild.wan_animate2":
        motion = analysis.get("caption") or "a person performing"
        return {
            **common,
            "reference_image": stage.input_rel(keyframe["path"]),
            "driving_video": clip_rel,
            "continue_video": prev_rel,
            "prompt": prompt,
            "video_frame_offset": w.overlap,
            "pose_prompt": f"{motion} Background stationary.",
        }
    if key in ("swap.scail2", "rebuild.scail2_animate"):
        return {
            **common,
            "reference_image": stage.input_rel(keyframe["path"]),
            "driving_video": clip_rel,
            "previous_video": prev_rel,
            "skip_frames": w.overlap,
            "video_frame_offset": w.overlap,
            "prompt": prompt,
            "max_people": max(1, len(analysis.get("people", []))),
        }
    if key == "swap.wan22_animate_mix":
        people = [p["track"] for p in analysis.get("people", [])]
        target = next(iter(decision.cast), "1")
        index = people.index(target) if target in people else 0
        return {
            **common,
            "reference_image": stage.input_rel(keyframe["path"]),
            "driving_video": clip_rel,
            "continue_video": prev_rel,
            "performer_index": str(index),
            "prompt": prompt,
            "video_frame_offset": w.overlap,
        }
    if key == "swap.bernini_rv2v":
        refs = [stage.input_rel(keyframe["path"])] if keyframe else []
        sheet = _character_sheet(stage, decision)
        if sheet:
            refs.append(sheet)
        text = shot.prompt or "Replace the person in the source video with the person from image0, keeping the motion."
        return {
            **common,
            "source_video": clip_rel,
            "reference_0": refs[0],
            "reference_1": refs[1] if len(refs) > 1 else None,
            "prompt": text,
        }
    if key == "restyle.bernini_v2v":
        return {
            **common,
            "source_video": clip_rel,
            "prompt": f"Restyle the video as {project.style}. {shot.prompt or ''}",
        }
    if key == "restyle.vace_fun":
        style_ref = stage.input_rel(keyframe["path"]) if keyframe else None
        if style_ref is None:
            raise ConfigError(f"{shot.id}: restyle.vace_fun needs a restyled keyframe", hint="higgsless keyframe make")
        return {
            **common,
            "source_video": clip_rel,
            "style_reference": style_ref,
            "prompt": f"{project.style}. {shot.prompt or ''}",
        }
    raise ConfigError(f"{key} is not a shot render engine")


def render_shot(
    stage: Stage, shot_id: str, *, takes: int | None = None, force: bool = False, gate: bool = True
) -> dict[str, Any]:
    stage.project.shot(shot_id)  # raises KeyError early for unknown shots
    decision = shot_decision(stage, shot_id)
    if decision.mode == Mode.PASSTHROUGH.value or not decision.engine:
        state = {"shot": shot_id, "mode": "passthrough", "takes": {}, "picked": None}
        stage.handle.state.write(f"renders/{shot_id}", state)
        return state
    if decision.mode == Mode.COMPOSITE.value:
        from higgsless.pipeline.composite import render_composite

        return render_composite(stage, shot_id, decision, takes=takes, force=force, gate=gate)
    return render_with(stage, shot_id, decision, takes=takes, force=force, gate=gate)


def render_with(
    stage: Stage, shot_id: str, decision: Decision, *, takes: int | None, force: bool, gate: bool
) -> dict[str, Any]:
    """Render all takes of a shot with the decision's engine, chaining windows."""
    project = stage.project
    shot = project.shot(shot_id)
    engine, warnings = registry().resolve(decision.mode, project, decision.engine)
    keyframe = None
    if decision.mode in NEEDS_KEYFRAME or engine.id == "restyle.vace_fun":
        keyframe = require_approved(stage, shot_id) if gate else stage.handle.state.read(f"keyframes/{shot_id}")
        if not keyframe:
            raise ConfigError(f"{shot_id} needs a keyframe", hint=f"higgsless keyframe make {shot_id}")
    width, height = gen_dims(stage, engine)
    clip, total = gen_clip(stage, shot_id, width, height)
    window = engine.constraints.window or length_4n1(total)
    windows = plan_windows(total, window=window, overlap=engine.constraints.overlap if total > window else 0)
    policy = stage.frame_policy()
    rate = format_rate(policy.gen_rate)
    n_takes = takes or shot.takes or project.takes
    runner = _runner(stage)
    previous = stage.handle.state.read(f"renders/{shot_id}") or {}
    out: dict[str, Any] = {
        "shot": shot_id,
        "mode": decision.mode,
        "engine": engine.id,
        "width": width,
        "height": height,
        "frames": total,
        "rate": rate,
        "decimate": policy.decimate,
        "windows": [w.to_dict() for w in windows],
        "takes": dict(previous.get("takes") or {}),
        "picked": previous.get("picked"),
        "warnings": warnings + decision.warnings,
    }
    for take in range(1, n_takes + 1):
        seed = int(hash_obj({"shot": shot_id, "take": take, "engine": engine.id}), 16) % (2**31)
        take_dir = stage.rpath("renders", shot_id, engine.id, f"take{take}")
        frames: list[str] = []
        prev_rel = None
        prev_key = None
        jobs = []
        for w in windows:
            win = window_clip(stage, shot_id, clip, total, w, width, height)
            params = engine.params_for(
                project.profile.value,
                engine_params(
                    stage, engine, decision, keyframe, w, stage.input_rel(win), prev_rel, width, height, seed
                ),
            )
            spec = JobSpec(
                engine,
                params,
                stage.join(take_dir, f"w{w.index:02d}"),
                f"render {shot_id} take{take} w{w.index}",
                inputs={"keyframe": (keyframe or {}).get("job", ""), "clip": win, "prev": prev_key or ""},
            )
            record = runner.run(spec, force=force)
            jobs.append(record["key"])
            produced = record["outputs"]["frames"]
            if len(produced) < w.emit_count:
                raise JobFailed(
                    f"{shot_id} take{take} window {w.index}: {len(produced)} frames, expected at least {w.emit_count}"
                )
            kept = produced[: w.emit_count]
            frames.extend(kept)
            if w is not windows[-1] and engine.constraints.overlap:
                tail = produced[-max(engine.constraints.overlap, 1) :]
                enc = stage.task(
                    "frames",
                    {"action": "encode", "paths": tail, "name": f"tail_w{w.index:02d}.mkv", "rate": rate},
                    take_dir,
                )
                prev_rel = stage.input_rel(enc["outputs"]["clip"][0])
                prev_key = record["key"]
        if len(frames) != total:
            raise JobFailed(f"{shot_id} take{take}: assembled {len(frames)} frames, expected {total}")
        listing = stage.task("frames", {"action": "write_list", "paths": frames, "name": "frames.json"}, take_dir)
        out["takes"][str(take)] = {
            "seed": seed,
            "frames_list": listing["outputs"]["list"][0],
            "count": len(frames),
            "jobs": jobs,
            "engine": engine.id,
            "reviewed": False,
        }
    if out["picked"] is None and out["takes"]:
        out["picked"] = "1"
    stage.handle.state.write(f"renders/{shot_id}", out)
    return out


def review_shot(stage: Stage, shot_id: str, *, force: bool = False) -> dict[str, Any]:
    state = stage.handle.state.read(f"renders/{shot_id}")
    if not state or not state.get("takes"):
        return {"shot": shot_id, "skipped": "nothing rendered"}
    clip = stage.join(stage.shot_dir(shot_id), f"gen_{state['width']}x{state['height']}_k{state['decimate']}.mkv")
    keyframe = stage.handle.state.read(f"keyframes/{shot_id}") or {}
    for take, info in state["takes"].items():
        if info.get("metrics") and not force:
            continue
        res = stage.task(
            "frames",
            {
                "action": "metrics",
                "list": info["frames_list"],
                "source_clip": clip,
                "keyframe": keyframe.get("path"),
                "expected": state["frames"],
                "sheet_name": f"review_{shot_id}_t{take}.jpg",
                "title": f"{shot_id} take {take} ({state['engine']})",
            },
            stage.rpath("previews"),
            gpu=True,
        )
        local = stage.pull(res["outputs"]["sheet"][0], f"review_{shot_id}_t{take}.jpg")
        info["metrics"] = {k: res[k] for k in ("frames", "flicker", "source_flicker", "flicker_ratio", "identity")}
        info["review_sheet"] = str(local)
        info["reviewed"] = True
    stage.handle.state.write(f"renders/{shot_id}", state)
    return state


def pick(stage: Stage, shot_id: str, take: int) -> dict[str, Any]:
    state = stage.handle.state.read(f"renders/{shot_id}")
    if not state or str(take) not in (state.get("takes") or {}):
        raise ConfigError(f"{shot_id} has no take {take}")
    state["picked"] = str(take)
    stage.handle.state.write(f"renders/{shot_id}", state)
    return state
