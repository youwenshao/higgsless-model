"""Shot analysis (SAM3 person tracks, faces, motion, captions) and cast clustering."""

from __future__ import annotations

from typing import Any

import numpy as np

from higgsless.analysis.cluster import cluster
from higgsless.core.manifest import Shot
from higgsless.engines.registry import registry
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import ConfigError, JobFailed
from higgsless.pipeline.stage import Stage

SHOT_CAPTION_PROMPT = (
    "Describe this music video frame for a video editor in one compact paragraph: shot size (extreme close-up, "
    "close-up, medium, wide), how many people and what they wear, the setting, lighting, and any visible camera "
    "movement cues. No speculation about story or emotions."
)


def _runner(stage: Stage) -> JobRunner:
    return JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, progress=stage.progress)


def analysis_clip(stage: Stage, shot: Shot) -> str:
    """480p generator-rate clip of the shot for tracking and captions."""
    path = stage.join(stage.shot_dir(shot.id), "analysis.mkv")
    if not stage.ex.exists(path):
        policy = stage.frame_policy()
        src = stage.require_source()
        width = int(round(src.width * 480 / min(src.width, src.height) / 16)) * 16
        height = int(round(src.height * 480 / min(src.width, src.height) / 16)) * 16
        stage.task(
            "extract",
            {
                "action": "gen_clip",
                "source": stage.join(stage.shot_dir(shot.id), "mezz.mp4"),
                "name": "analysis.mkv",
                "width": width,
                "height": height,
                "decimate": policy.decimate,
            },
            stage.shot_dir(shot.id),
        )
    return path


def track(stage: Stage, shot: Shot, *, max_people: int = 8, force: bool = False) -> str:
    """SAM 3.1 person tracking rendered as colored identity masks (one PNG per frame)."""
    engine, _ = registry().resolve("track", stage.project)
    clip = analysis_clip(stage, shot)
    out_dir = stage.join(stage.shot_dir(shot.id), "track")
    params = engine.params_for(
        stage.project.profile.value,
        {
            "video": stage.input_rel(clip),
            "text": "person",
            "max_objects": max_people,
        },
    )
    record = _runner(stage).run(JobSpec(engine, params, out_dir, f"track {shot.id}"), force=force)
    return stage.join(out_dir, "masks") if record["outputs"].get("masks") else out_dir


def caption(stage: Stage, shot: Shot, image_path: str, prompt: str, *, label: str, force: bool = False) -> str:
    engine, _ = registry().resolve("caption", stage.project)
    out_dir = stage.join(stage.shot_dir(shot.id), "captions", label)
    params = engine.params_for(stage.project.profile.value, {"image": stage.input_rel(image_path), "prompt": prompt})
    record = _runner(stage).run(JobSpec(engine, params, out_dir, f"caption {shot.id} {label}"), force=force)
    texts = record.get("outputs", {}).get("text") or []
    if not texts:
        raise JobFailed(f"caption {shot.id}: no text produced")
    return stage.ex.read_text(texts[0]).strip()


def analyze_shot(stage: Stage, shot: Shot, *, captions: bool = True, force: bool = False) -> dict[str, Any]:
    mask_dir = track(stage, shot, force=force)
    clip = analysis_clip(stage, shot)
    out_dir = stage.join(stage.shot_dir(shot.id), "analysis")
    result = stage.task(
        "analyze",
        {"video": clip, "mask_dir": mask_dir, "mezz": stage.join(stage.shot_dir(shot.id), "mezz.mp4")},
        out_dir,
    )
    data = {k: result[k] for k in ("people", "motion", "faces")}
    data["mask_dir"] = mask_dir
    if captions:
        first = stage.join(stage.shot_dir(shot.id), "first.png")
        data["caption"] = caption(stage, shot, first, SHOT_CAPTION_PROMPT, label="shot", force=force)
    stage.handle.state.write(f"analysis/{shot.id}", data)
    return data


def analyze(stage: Stage, shot_ids: list[str] | None, *, captions: bool = True, force: bool = False) -> dict:
    project = stage.project
    if not project.shots:
        raise ConfigError("no shots yet", hint="higgsless shots detect")
    shots = [project.shot(i) for i in shot_ids] if shot_ids else project.shots
    done = {}
    for shot in shots:
        if not force and stage.handle.state.exists(f"analysis/{shot.id}"):
            done[shot.id] = stage.handle.state.read(f"analysis/{shot.id}")
            continue
        stage.progress(f"analysing {shot.id}")
        done[shot.id] = analyze_shot(stage, shot, captions=captions, force=force)
    return {
        sid: {"people": len(d["people"]), "faces": d["faces"].get("max_frac"), "motion": d["motion"]}
        for sid, d in done.items()
    }


def detect_cast(stage: Stage, *, threshold: float = 0.72) -> dict[str, Any]:
    """Cluster person crops across shots into performers P1, P2, ..."""
    members: list[dict[str, Any]] = []
    for shot in stage.project.shots:
        data = stage.handle.state.read(f"analysis/{shot.id}")
        if not data:
            raise ConfigError(f"{shot.id} is not analysed yet", hint="higgsless analyze")
        for person in data["people"]:
            if person.get("crop"):
                members.append(
                    {"shot": shot.id, "track": person["track"], "crop": person["crop"], "presence": person["presence"]}
                )
    if not members:
        stage.handle.state.write("cast", {"performers": {}, "method": None})
        return {"performers": {}}
    res = stage.task("embed", {"paths": [m["crop"] for m in members]}, stage.rpath("analysis", "cast"), gpu=True)
    by_path = dict(zip(res["paths"], res["embeddings"], strict=True))
    members = [m for m in members if m["crop"] in by_path]
    vectors = np.array([by_path[m["crop"]] for m in members], dtype=np.float32)
    labels = cluster(vectors, threshold if res["method"] != "histogram" else 0.9)
    performers: dict[str, dict[str, Any]] = {}
    for member, label in zip(members, labels, strict=True):
        pid = f"P{label + 1}"
        performers.setdefault(pid, {"members": []})["members"].append(
            {"shot": member["shot"], "track": member["track"], "crop": member["crop"]}
        )
    for perf in performers.values():
        perf["shots"] = sorted({m["shot"] for m in perf["members"]})

    items = []
    for pid, perf in performers.items():
        for m in perf["members"][:6]:
            items.append({"path": m["crop"], "label": f"{pid} {m['shot']}#{m['track']}"})
    sheet = stage.task(
        "contact",
        {"items": items, "name": "cast.jpg", "title": "performers", "cell_width": 200},
        stage.rpath("previews"),
    )
    local = stage.pull(sheet["outputs"]["sheet"][0], "cast.jpg")
    state = {"performers": performers, "method": res["method"], "threshold": threshold, "contact_sheet": str(local)}
    stage.handle.state.write("cast", state)
    return state


def performer_of(stage: Stage, shot_id: str, track: str) -> str | None:
    cast = stage.handle.state.read("cast") or {}
    for pid, perf in (cast.get("performers") or {}).items():
        if any(m["shot"] == shot_id and m["track"] == track for m in perf["members"]):
            return pid
    return None
