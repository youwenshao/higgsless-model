"""Characters (portrait, sheet, angles) and per-shot sample-scene keyframes with a review gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.core.manifest import Mode
from higgsless.engines.registry import registry
from higgsless.engines.router import Decision, decide
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import ConfigError, GateBlocked, ValidationError
from higgsless.pipeline.analyze import caption, performer_of
from higgsless.pipeline.stage import Stage

SHEET_PROMPT = (
    "A character reference sheet of the character in image 1 on a clean white seamless background, landscape: "
    "full-body front view, full-body side view, full-body back view, and a large face close-up. The same face, "
    "hairstyle, outfit and body proportions in every panel. Soft, even studio light. No text, no labels."
)
PORTRAIT_PROMPT = (
    "Full-body portrait of {prompt}, standing in a relaxed pose facing the camera, on a plain light-grey studio "
    "background, soft even light, natural proportions, head and shoes in frame. {style}"
)
ANGLE_PROMPTS = {
    "left": "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
    "right": "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
    "back": "将镜头旋转180度 Rotate the camera 180 degrees to show the back.",
}
CAPTION_PROMPTS = {
    "en": (
        "Describe this image objectively in exactly two lines and nothing else.\n"
        "Line 1 starts with 'Character Description:' and describes each person's appearance and clothing "
        "(no actions).\nLine 2 starts with 'Background description:' and describes the setting and lighting."
    ),
    "zh": (
        "用中文客观描述图片中的内容，包括以下要点：人物外观描述，不描述动作行为。背景描述，忽略主观评价和情绪推测。"
        "下面给出描述范例，必须遵循这个范式，不要输出额外的符号：人物外观描述：穿着一件浅蓝色的校服衬衫。背景描述：背景为明亮、整洁的教室。"
    ),
}
POSITIONS = ("on the left", "in the centre", "on the right")


def _runner(stage: Stage) -> JobRunner:
    return JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, progress=stage.progress)


def _first_output(record: dict[str, Any], name: str) -> str:
    paths = (record.get("outputs") or {}).get(name) or []
    if not paths:
        raise ValidationError(f"job {record.get('label')} produced no {name}")
    return paths[0]


# Characters ------------------------------------------------------------------------------------
def upload_refs(stage: Stage, name: str, images: list[Path]) -> list[str]:
    paths = []
    for i, image in enumerate(images, start=1):
        if not image.exists():
            raise ConfigError(f"reference image not found: {image}")
        remote = stage.rpath("characters", name, "refs", f"ref_{i}{image.suffix.lower()}")
        stage.ex.makedirs(stage.rpath("characters", name, "refs"))
        stage.ex.put(image, remote)
        paths.append(f"characters/{name}/refs/ref_{i}{image.suffix.lower()}")
    return paths


def create_character(
    stage: Stage,
    name: str,
    *,
    prompt: str | None,
    images: list[Path],
    style: str | None,
    stylised: bool | None,
    angles: bool,
    seed: int | None,
    force: bool,
) -> dict[str, Any]:
    with stage.handle.edit() as data:
        entry = data["characters"].setdefault(name, {})
        if prompt is not None:
            entry["prompt"] = prompt
        if style is not None:
            entry["style"] = style
        if stylised is not None:
            entry["stylised"] = stylised
        if images:
            entry["images"] = upload_refs(stage, name, images)
    character = stage.project.characters[name]
    if not character.images and not character.prompt:
        raise ValidationError(f"character {name} needs --prompt or --image")
    project = stage.project
    runner = _runner(stage)
    root = stage.rpath("characters", name)
    result: dict[str, Any] = {"character": name}

    anchor = stage.rpath(character.images[0]) if character.images else None
    if anchor is None:
        engine, _ = registry().resolve("portrait", project)
        params = engine.params_for(
            project.profile.value,
            {"prompt": PORTRAIT_PROMPT.format(prompt=character.prompt, style=character.style or ""), "seed": seed},
        )
        record = runner.run(JobSpec(engine, params, stage.join(root, "portrait"), f"portrait {name}"), force=force)
        anchor = _first_output(record, "image")
        result["portrait"] = anchor

    engine, _ = registry().resolve("sheet", project)
    extra = {f"image_{i}": stage.input_rel(stage.rpath(p)) for i, p in enumerate(character.images[1:4], start=2)}
    prompt_text = SHEET_PROMPT + (f" Character: {character.prompt}." if character.prompt else "")
    if character.style:
        prompt_text += f" Style: {character.style}."
    params = engine.params_for(
        project.profile.value, {"prompt": prompt_text, "image_1": stage.input_rel(anchor), "seed": seed, **extra}
    )
    record = runner.run(JobSpec(engine, params, stage.join(root, "sheet"), f"sheet {name}"), force=force)
    sheet = _first_output(record, "image")
    result["sheet"] = sheet

    views = []
    if angles:
        kf_engine, _ = registry().resolve("keyframe", project)
        for view, text in ANGLE_PROMPTS.items():
            params = kf_engine.params_for(
                project.profile.value,
                {
                    "base_image": stage.input_rel(anchor),
                    "prompt": text,
                    "seed": seed,
                    "angle_lora": "Qwen-Edit-2509-Multiple-angles.safetensors",
                },
            )
            rec = runner.run(
                JobSpec(kf_engine, params, stage.join(root, "angles", view), f"{name} {view} view"), force=force
            )
            views.append(_first_output(rec, "image"))
        result["angles"] = views

    with stage.handle.edit() as data:
        data["characters"][name]["sheet"] = stage.input_rel(sheet).removeprefix(f"projects/{project.project}/")
    items = [{"path": anchor, "label": f"{name}: anchor"}, {"path": sheet, "label": f"{name}: sheet"}]
    items += [{"path": v, "label": f"{name}: {k}"} for k, v in zip(ANGLE_PROMPTS, views, strict=False)]
    res = stage.task(
        "contact",
        {"items": items, "name": f"character_{name}.jpg", "title": f"character {name}", "cell_width": 420},
        stage.rpath("previews"),
    )
    result["contact_sheet"] = str(stage.pull(res["outputs"]["sheet"][0], f"character_{name}.jpg"))
    stage.handle.state.write(f"characters/{name}", result)
    return result


# Keyframes -------------------------------------------------------------------------------------
def shot_decision(stage: Stage, shot_id: str) -> Decision:
    project = stage.project
    shot = project.shot(shot_id)
    analysis = stage.handle.state.read(f"analysis/{shot.id}")
    tracks = {}
    for person in (analysis or {}).get("people", []):
        perf = performer_of(stage, shot.id, person["track"])
        if perf:
            tracks[person["track"]] = perf
    return decide(project, shot, analysis, registry(), tracks, new_vocals=bool(project.audio.lead_voice))


def _position(person: dict[str, Any]) -> str:
    x0, _, x1, _ = person.get("first_box") or [0.4, 0, 0.6, 1]
    centre = (x0 + x1) / 2
    return POSITIONS[0] if centre < 0.38 else POSITIONS[2] if centre > 0.62 else POSITIONS[1]


def keyframe_prompt(stage: Stage, shot_id: str, decision: Decision, refs: list[str]) -> str:
    project = stage.project
    shot = project.shot(shot_id)
    analysis = stage.handle.state.read(f"analysis/{shot_id}") or {}
    people = {p["track"]: p for p in analysis.get("people", [])}
    keep = "Keep the pose, expression, framing, camera angle"
    if decision.mode == Mode.RESTYLE.value:
        return f"Restyle this image as {project.style}. Keep the composition, poses and framing exactly the same."
    parts = []
    for i, (track, _) in enumerate(decision.cast.items()):
        where = _position(people[track]) if track in people else ""
        parts.append(f"replace the person {where} with the character from image {i + 2}".replace("  ", " "))
    instruction = "; ".join(parts) or "replace the person with the character from image 2"
    text = instruction[0].upper() + instruction[1:] + ". "
    if decision.mode == Mode.REBUILD.value and project.world:
        text += f"Change the setting to {project.world}. {keep} and lighting direction exactly the same."
    else:
        text += f"{keep}, background and lighting exactly the same."
    if shot.prompt:
        text += f" {shot.prompt}"
    return text


def make_keyframe(stage: Stage, shot_id: str, *, seed: int | None, force: bool) -> dict[str, Any]:
    project = stage.project
    decision = shot_decision(stage, shot_id)
    if decision.mode == Mode.PASSTHROUGH.value:
        return {"shot": shot_id, "skipped": "passthrough"}
    refs = []
    for character in dict.fromkeys(decision.cast.values()):
        sheet = project.characters[character].sheet
        if not sheet:
            raise ConfigError(f"character {character} has no sheet yet", hint=f"higgsless char create {character}")
        refs.append(stage.input_rel(stage.rpath(sheet)))
    if decision.mode != Mode.RESTYLE.value and not refs:
        raise ConfigError(f"{shot_id}: nobody is cast", hint="higgsless cast assign <performer> <character>")
    engine, _ = registry().resolve("keyframe", project)
    prompt = keyframe_prompt(stage, shot_id, decision, refs)
    params = engine.params_for(
        project.profile.value,
        {
            "base_image": stage.input_rel(stage.join(stage.shot_dir(shot_id), "first.png")),
            "reference_2": refs[0] if refs else None,
            "reference_3": refs[1] if len(refs) > 1 else None,
            "prompt": prompt,
            "seed": seed,
        },
    )
    out = stage.rpath("keyframes", shot_id)
    record = _runner(stage).run(JobSpec(engine, params, out, f"keyframe {shot_id}"), force=force)
    keyframe = _first_output(record, "image")
    caption_text = caption(
        stage, project.shot(shot_id), keyframe, CAPTION_PROMPTS[project.caption_language], label="keyframe", force=force
    )
    previous = stage.handle.state.read(f"keyframes/{shot_id}") or {}
    state = {
        "shot": shot_id,
        "path": keyframe,
        "prompt": prompt,
        "caption": caption_text,
        "mode": decision.mode,
        "engine": decision.engine,
        "cast": decision.cast,
        "job": record["key"],
        "approved": previous.get("approved", False) and previous.get("job") == record["key"],
    }
    stage.handle.state.write(f"keyframes/{shot_id}", state)
    return state


def keyframes_contact_sheet(stage: Stage) -> str | None:
    items = []
    for shot in stage.project.shots:
        state = stage.handle.state.read(f"keyframes/{shot.id}")
        if not state or "path" not in state:
            continue
        items.append({"path": stage.join(stage.shot_dir(shot.id), "first.png"), "label": f"{shot.id} original"})
        mark = "approved" if state.get("approved") else "pending"
        items.append({"path": state["path"], "label": f"{shot.id} keyframe ({mark})"})
    if not items:
        return None
    res = stage.task(
        "contact",
        {"items": items, "name": "keyframes.jpg", "title": "keyframes", "columns": 4},
        stage.rpath("previews"),
    )
    return str(stage.pull(res["outputs"]["sheet"][0], "keyframes.jpg"))


def set_approval(stage: Stage, shot_ids: list[str], approved: bool, note: str | None = None) -> list[str]:
    changed = []
    for shot_id in shot_ids:
        state = stage.handle.state.read(f"keyframes/{shot_id}")
        if not state or "path" not in state:
            continue
        state["approved"] = approved
        if note:
            state["note"] = note
        stage.handle.state.write(f"keyframes/{shot_id}", state)
        changed.append(shot_id)
    return changed


def require_approved(stage: Stage, shot_id: str) -> dict[str, Any]:
    state = stage.handle.state.read(f"keyframes/{shot_id}")
    if not state or "path" not in state:
        raise GateBlocked(f"{shot_id} has no keyframe yet", hint=f"higgsless keyframe make {shot_id}")
    if not state.get("approved"):
        raise GateBlocked(
            f"{shot_id}'s keyframe is not approved",
            hint=f"review previews/keyframes.jpg, then higgsless keyframe approve {shot_id}",
            details={"keyframe": state["path"], "prompt": state["prompt"]},
        )
    return state
