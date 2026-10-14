"""Run the stages that are ready, in order, and report where a project stands."""

from __future__ import annotations

from typing import Any

from higgsless.errors import ConfigError, GateBlocked, ValidationError
from higgsless.pipeline import audio as audio_stage
from higgsless.pipeline import finish as finish_stage
from higgsless.pipeline import lipsync as lipsync_stage
from higgsless.pipeline import render as render_stage
from higgsless.pipeline.conform import conform
from higgsless.pipeline.images import keyframes_contact_sheet, make_keyframe
from higgsless.pipeline.stage import Stage

ORDER = ["analyze", "keyframes", "render", "audio", "lipsync", "finish", "conform"]


def status(stage: Stage) -> dict[str, Any]:
    project = stage.project
    handle = stage.handle
    audio = handle.state.read("audio") or {}
    rendered = [s.id for s in project.shots if (handle.state.read(f"renders/{s.id}") or {}).get("takes")]
    approved = [s.id for s in project.shots if (handle.state.read(f"keyframes/{s.id}") or {}).get("approved")]
    return {
        "project": project.project,
        "shots": len(project.shots),
        "analysed": sum(1 for s in project.shots if handle.state.exists(f"analysis/{s.id}")),
        "keyframes_approved": approved,
        "rendered": rendered,
        "audio": {
            "separated": "stems" in audio,
            "transcribed": "metadata" in audio,
            "sung": "lead_new" in audio,
            "mixed": "final_mix" in audio,
        },
        "lipsynced": sum(1 for s in project.shots if handle.state.exists(f"lipsync/{s.id}")),
        "finished": sum(1 for s in project.shots if handle.state.exists(f"finish/{s.id}")),
        "conformed": bool(handle.state.read("conform")),
    }


def _shots(stage: Stage) -> list[str]:
    if not stage.project.shots:
        raise ConfigError("no shots", hint="higgsless shots detect")
    return [s.id for s in stage.project.shots]


def run(stage: Stage, until: str) -> dict[str, Any]:
    if until not in ORDER:
        raise ConfigError(f"unknown stage {until!r}", hint=f"one of {ORDER}")
    done: dict[str, Any] = {}
    for step in ORDER:
        stage.progress(f"stage: {step}")
        if step == "analyze":
            from higgsless.pipeline.analyze import analyze

            done[step] = analyze(stage, None)
        elif step == "keyframes":
            results = {sid: make_keyframe(stage, sid, seed=None, force=False) for sid in _shots(stage)}
            pending = [sid for sid, r in results.items() if r.get("path") and not r.get("approved")]
            done[step] = {"contact_sheet": keyframes_contact_sheet(stage), "pending": pending}
            if pending:
                raise GateBlocked(
                    f"{len(pending)} keyframe(s) need approval",
                    hint="review previews/keyframes.jpg, then higgsless keyframe approve --all",
                    details=done,
                )
        elif step == "render":
            done[step] = {sid: render_stage.render_shot(stage, sid) for sid in _shots(stage)}
        elif step == "audio":
            done[step] = _audio(stage)
        elif step == "lipsync":
            done[step] = {sid: lipsync_stage.lipsync_shot(stage, sid) for sid in _shots(stage)}
        elif step == "finish":
            done[step] = {sid: finish_stage.finish_shot(stage, sid) for sid in _shots(stage)}
        elif step == "conform":
            done[step] = conform(stage, pull=True)
        if step == until:
            break
    return done


def _audio(stage: Stage) -> dict[str, Any]:
    separated = audio_stage.separate(stage)
    if stage.project.audio.lead_voice is None:
        raise ConfigError("no lead voice yet", hint="higgsless voice create <name> --kind designed --describe '...'")
    transcribed = audio_stage.transcribe(stage)
    report = audio_stage.fit_lyrics(stage)
    if not report["ok"]:
        raise ValidationError(
            "the lyrics do not fit the melody", details=report, hint=f"edit {transcribed['lyrics_file']} and rerun"
        )
    return {
        "stems": list(separated["stems"]),
        "lead_new": audio_stage.sing(stage)["lead_new"],
        "final_mix": audio_stage.mix(stage)["final_mix"],
    }
