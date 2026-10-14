"""The `auto` router: pick mode, engine and lip-sync per shot, with reasons agents can read."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from higgsless.core.manifest import Look, Mode, Project, Shot, Switch
from higgsless.engines.registry import Registry

MIN_PRESENCE = 0.2  # tracked people visible in less of the shot than this are ignored
LIPSYNC_MIN_FACE = 0.10  # face height / frame height below which lip-sync is skipped

DEFAULTS = {
    (Mode.SWAP, "single"): "swap.wan22_animate_mix",
    (Mode.SWAP, "multi"): "swap.scail2",
    (Mode.REBUILD, "single"): "rebuild.wan_animate2",
    (Mode.REBUILD, "multi"): "rebuild.scail2_animate",
    (Mode.RESTYLE, "single"): "restyle.bernini_v2v",
    (Mode.RESTYLE, "multi"): "restyle.bernini_v2v",
    (Mode.COMPOSITE, "single"): "composite.animate2_matte",
    (Mode.COMPOSITE, "multi"): "composite.animate2_matte",
}


@dataclass
class Decision:
    shot: str
    mode: str
    engine: str | None
    cast: dict[str, str] = field(default_factory=dict)  # track -> character
    lipsync: bool = False
    lipsync_engine: str | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_cast(
    project: Project, shot: Shot, analysis: dict[str, Any] | None, performer_of: dict[str, str]
) -> dict[str, str]:
    """track -> character, from the shot's own cast, else the performer's global casting."""
    cast: dict[str, str] = {}
    for person in (analysis or {}).get("people", []):
        if person.get("presence", 1.0) < MIN_PRESENCE:
            continue
        track = person["track"]
        performer = performer_of.get(track)
        character = shot.cast.get(track) or (shot.cast.get(performer) if performer else None)
        character = character or (project.cast.get(performer) if performer else None)
        if character:
            cast[track] = character
    if not analysis:
        cast.update({k: v for k, v in shot.cast.items() if k.isdigit()})
    return cast


def decide(
    project: Project,
    shot: Shot,
    analysis: dict[str, Any] | None,
    registry: Registry,
    performer_of: dict[str, str] | None = None,
    new_vocals: bool = True,
) -> Decision:
    d = Decision(shot=shot.id, mode=Mode.PASSTHROUGH.value, engine=None)
    people = [p for p in (analysis or {}).get("people", []) if p.get("presence", 1.0) >= MIN_PRESENCE]
    cast = resolve_cast(project, shot, analysis, performer_of or {})
    d.cast = cast

    if shot.mode is not Mode.AUTO:
        mode = shot.mode
        d.reasons.append(f"mode {mode.value} set on the shot")
    elif project.look is Look.RESTYLE:
        mode = Mode.RESTYLE
        d.reasons.append("project look is restyle")
    elif analysis is not None and not people:
        mode = Mode.PASSTHROUGH
        d.reasons.append("no people tracked in this shot")
    elif analysis is not None and not cast:
        mode = Mode.PASSTHROUGH
        d.reasons.append("nobody in this shot is cast as a character")
        d.warnings.append("cast performers with `cast assign` or set shot cast to replace them")
    elif project.look is Look.ORIGINAL:
        mode = Mode.SWAP
        d.reasons.append("project look is original: keep the set, swap performers")
    else:
        mode = Mode.REBUILD
        d.reasons.append("project look is new-world: rebuild the shot around the characters")
    d.mode = mode.value

    if mode is not Mode.PASSTHROUGH:
        count = len(cast) if cast else len(people)
        size = "multi" if count >= 2 else "single"
        if shot.engine:
            engine_id = shot.engine
            d.reasons.append(f"engine {engine_id} set on the shot")
        elif mode.value in project.engines:
            engine_id = project.engines[mode.value]
            d.reasons.append(f"project default engine for {mode.value}")
        else:
            engine_id = DEFAULTS[(mode, size)]
            d.reasons.append(f"{count} person(s) to replace -> {engine_id}")
        engine = registry.get(engine_id)
        limit = engine.constraints.max_people
        if limit is not None and count > limit:
            d.warnings.append(f"{engine_id} handles up to {limit} people; this shot has {count}")
        d.engine = engine_id
        if len(people) > len(cast) and mode is Mode.SWAP:
            d.warnings.append(f"{len(people) - len(cast)} tracked person(s) are not cast and stay as filmed")

    faces = (analysis or {}).get("faces") or {}
    face = faces.get("max_frac")
    if shot.lipsync is Switch.OFF:
        d.lipsync = False
        d.reasons.append("lip-sync off on the shot")
    elif shot.lipsync is Switch.ON:
        d.lipsync = True
        d.reasons.append("lip-sync on for the shot")
    elif not new_vocals:
        d.reasons.append("no new vocals: lip-sync not needed")
    elif face is None:
        d.warnings.append("face size unknown (run analyze); lip-sync decided at render time")
    elif face >= LIPSYNC_MIN_FACE:
        d.lipsync = True
        d.reasons.append(f"face fills {face:.0%} of the frame height: lip-sync on")
    else:
        d.reasons.append(f"largest face is {face:.0%} of the frame: lip-sync skipped")
    if d.lipsync:
        stylised = any(project.characters.get(c) and project.characters[c].stylised for c in cast.values())
        default = "lipsync.infinitetalk" if stylised else "lipsync.latentsync16"
        d.lipsync_engine = project.engines.get("lipsync", default)
    return d
