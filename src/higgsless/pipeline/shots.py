"""Shot detection, per-shot media, and manual split/merge."""

from __future__ import annotations

import string
from typing import Any

from higgsless.core.manifest import Shot
from higgsless.core.timecode import frame_to_timecode, parse_rate
from higgsless.errors import ConfigError, ValidationError
from higgsless.pipeline.stage import Stage

_CARRY = ("mode", "engine", "cast", "lipsync", "prompt", "takes", "notes", "transition")


def _ingest_state(stage: Stage) -> dict[str, Any]:
    state = stage.handle.state.read("ingest")
    if not state:
        raise ConfigError("ingest has not run", hint="higgsless ingest <video>")
    return state


def shot_entries(detected: list[dict[str, int]], transitions: list[dict[str, Any]], old: list[Shot]) -> list[dict]:
    """Detected [start, end) ranges -> manifest entries, keeping settings of unchanged shots."""
    previous = {(s.start, s.end): s for s in old}
    entries = []
    for i, shot in enumerate(detected, start=1):
        entry: dict[str, Any] = {"id": f"s{i:03d}", "frames": [shot["start"], shot["end"]]}
        for t in transitions:
            if t.get("kind") == "dissolve" and abs(int(t["frame"]) - shot["start"]) <= 2 and shot["start"] > 0:
                entry["transition"] = f"dissolve:{int(t.get('length', 8))}"
        prev = previous.get((shot["start"], shot["end"]))
        if prev is not None:
            defaults = Shot(id="s001", frames=[0, 1])
            for key in _CARRY:
                value = getattr(prev, key)
                if value != getattr(defaults, key):
                    entry[key] = value.value if hasattr(value, "value") else value
        entries.append(entry)
    return entries


def extract_shots(stage: Stage, shots: list[Shot]) -> dict[str, Any]:
    source = stage.require_source()
    ingest_state = _ingest_state(stage)
    payload = [{"id": s.id, "start": s.start, "end": s.end} for s in shots]
    stage.progress(f"cutting {len(payload)} shot mezzanine(s)")
    return stage.task(
        "extract",
        {"action": "shots", "source": ingest_state["remote_source"], "rate": source.rate, "shots": payload},
        stage.rpath("shots"),
    )["outputs"]


def shots_contact_sheet(stage: Stage) -> str | None:
    project = stage.project
    rate = parse_rate(project.source.rate) if project.source else None
    items = []
    for s in project.shots:
        label = f"{s.id} {frame_to_timecode(s.start, rate) if rate else s.start} ({s.length}f)"
        items.append({"path": stage.join(stage.shot_dir(s.id), "first.png"), "label": label})
    if not items:
        return None
    res = stage.task("contact", {"items": items, "name": "shots.jpg", "title": f"{project.project}: shots"},
                     stage.rpath("previews"))
    return str(stage.pull(res["outputs"]["sheet"][0], "shots.jpg"))


def detect(stage: Stage, *, min_frames: int = 8, adaptive: float = 3.0, content: float = 27.0,
           transnet: bool = True, force: bool = False) -> dict[str, Any]:
    project = stage.project
    source = stage.require_source()
    if project.shots and not force:
        raise ConfigError("shots are already detected", hint="pass --force to redo; settings of unchanged shots are kept")
    ingest_state = _ingest_state(stage)
    stage.progress("detecting shot boundaries")
    res = stage.task(
        "shots",
        {"path": ingest_state["proxy"], "frames": source.frames, "min_frames": min_frames, "adaptive": adaptive,
         "content": content, "transnet": transnet},
        stage.rpath("shots"),
    )
    entries = shot_entries(res["shots"], res.get("transitions", []), project.shots)
    with stage.handle.edit() as data:
        data["shots"] = entries
    extract_shots(stage, stage.project.shots)
    sheet = shots_contact_sheet(stage)
    summary = {"count": len(entries), "detectors": res.get("detectors"), "transitions": res.get("transitions", []),
               "contact_sheet": sheet}
    stage.handle.state.write("shots", summary)
    return summary


def _next_suffix(existing: set[str], base: str) -> str:
    for letter in string.ascii_lowercase[1:]:
        candidate = f"{base}{letter}"
        if candidate not in existing:
            return candidate
    raise ValidationError(f"too many splits of {base}")


def split(stage: Stage, shot_id: str, at: int) -> list[str]:
    project = stage.project
    shot = project.shot(shot_id)
    if not shot.start < at < shot.end:
        raise ValidationError(f"split frame {at} is not inside {shot_id} [{shot.start}, {shot.end})")
    base = shot_id.rstrip(string.ascii_lowercase)
    new_id = _next_suffix({s.id for s in project.shots}, base)
    with stage.handle.edit() as data:
        for i, entry in enumerate(data["shots"]):
            if entry["id"] == shot_id:
                entry["frames"] = [shot.start, at]
                data["shots"].insert(i + 1, {"id": new_id, "frames": [at, shot.end]})
                break
    project = stage.project
    extract_shots(stage, [project.shot(shot_id), project.shot(new_id)])
    return [shot_id, new_id]


def merge(stage: Stage, first: str, second: str) -> str:
    project = stage.project
    ids = [s.id for s in project.shots]
    a, b = project.shot(first), project.shot(second)
    if ids.index(second) != ids.index(first) + 1:
        raise ValidationError(f"{first} and {second} are not adjacent")
    with stage.handle.edit() as data:
        data["shots"] = [e for e in data["shots"] if e["id"] != second]
        for entry in data["shots"]:
            if entry["id"] == first:
                entry["frames"] = [a.start, b.end]
    extract_shots(stage, [stage.project.shot(first)])
    return first
