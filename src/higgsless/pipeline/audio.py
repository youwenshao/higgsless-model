"""Audio stages: separate, transcribe, fit lyrics, design or load a voice, sing, mix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from higgsless.audio.syllables import fit, segment_text
from higgsless.core.manifest import Consent, VoiceKind
from higgsless.engines.registry import registry
from higgsless.engines.runner import JobRunner, JobSpec
from higgsless.errors import ConfigError, ValidationError
from higgsless.pipeline.stage import Stage

SOULX = {"en", "zh", "yue"}


def _runner(stage: Stage) -> JobRunner:
    return JobRunner(stage.cfg, stage.ex, stage.app.comfy, stage.handle.state, progress=stage.progress)


def _run(
    stage: Stage, engine_id: str, params: dict[str, Any], output_dir: str, label: str, force: bool = False
) -> dict:
    engine = registry().get(engine_id)
    merged = engine.params_for(stage.project.profile.value, params)
    return _runner(stage).run(JobSpec(engine, merged, output_dir, label), force=force)


def _first(record: dict, name: str) -> str:
    paths = (record.get("outputs") or {}).get(name) or []
    if not paths:
        raise ValidationError(f"{record.get('label')} produced no {name}")
    return paths[0]


def separate(stage: Stage, *, force: bool = False) -> dict[str, Any]:
    audio = (stage.handle.state.read("ingest") or {}).get("audio")
    if not audio:
        raise ConfigError("no original audio", hint="higgsless ingest <video> on a file that has audio")
    record = _run(stage, "separate.roformer", {"source": audio}, stage.rpath("audio", "stems"), "separate", force)
    stems = {name: paths[0] for name, paths in record["outputs"].items()}
    state = {**(stage.handle.state.read("audio") or {}), "stems": stems}
    stage.handle.state.write("audio", state)
    return state


def transcribe(stage: Stage, *, force: bool = False) -> dict[str, Any]:
    stems = (stage.handle.state.read("audio") or {}).get("stems") or {}
    if "lead_dry" not in stems:
        raise ConfigError("no dry lead vocal", hint="higgsless audio separate")
    language = stage.project.audio.language
    record = _run(
        stage,
        "transcribe.soulx_pre",
        {"action": "preprocess", "audio": stems["lead_dry"], "language": language},
        stage.rpath("audio", "transcription"),
        "transcribe",
        force,
    )
    remote = _first(record, "metadata")
    local = stage.handle.root / "audio"
    local.mkdir(exist_ok=True)
    stage.ex.get(remote, local / "metadata.json")
    data = json.loads((local / "metadata.json").read_text(encoding="utf-8"))
    preview = local / "lyrics_original.txt"
    preview.write_text("\n".join(segment_text(s) for s in data) + "\n", encoding="utf-8")
    template = local / stage.project.audio.lyrics_new
    if not template.exists():
        yaml = YAML()
        payload = {"language": language, "lines": [segment_text(s) for s in data]}
        with template.open("w", encoding="utf-8") as fh:
            yaml.dump(payload, fh)
    state = {**(stage.handle.state.read("audio") or {}), "metadata": remote, "segments": len(data)}
    stage.handle.state.write("audio", state)
    return {**state, "lyrics_file": str(template), "original": str(preview)}


def fit_lyrics(stage: Stage) -> dict[str, Any]:
    meta_path = stage.handle.root / "audio" / "metadata.json"
    if not meta_path.exists():
        raise ConfigError("no transcription", hint="higgsless audio transcribe")
    lyrics_path = stage.handle.root / "audio" / stage.project.audio.lyrics_new
    if not lyrics_path.exists():
        raise ConfigError(f"missing {lyrics_path.name}", hint="higgsless audio transcribe writes a starter file")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    doc = YAML(typ="safe").load(lyrics_path.read_text(encoding="utf-8")) or {}
    language = str(doc.get("language") or stage.project.audio.language)
    return fit(metadata, list(doc.get("lines") or []), language)


def _voice(stage: Stage) -> tuple[str, Any]:
    project = stage.project
    name = project.audio.lead_voice
    if not name or name not in project.voices:
        raise ConfigError("no lead voice", hint="higgsless voice create <name> ... then set audio.lead_voice")
    return name, project.voices[name]


def design_voice(stage: Stage, name: str, describe: str, *, text: str | None, force: bool) -> dict[str, Any]:
    record = _run(
        stage,
        "voice.voxcpm2_design",
        {"action": "design", "describe": describe, "text": text},
        stage.rpath("audio", "voices", name),
        f"voice {name}",
        force,
    )
    reference = _first(record, "reference")
    with stage.handle.edit() as data:
        data["voices"][name] = {
            "kind": "designed",
            "consent": "synthetic",
            "describe": describe,
            "reference": reference,
            "language": data["audio"]["language"],
        }
        data["audio"]["lead_voice"] = name
    return {"voice": name, "reference": reference}


def sing(stage: Stage, *, force: bool = False) -> dict[str, Any]:
    report = fit_lyrics(stage)
    if not report["ok"]:
        raise ValidationError(
            "the new lyrics do not fit the melody",
            details=report,
            hint="edit audio/lyrics_new.yaml so each line has one word per sung note",
        )
    name, voice = _voice(stage)
    language = stage.project.audio.language
    lyrics = YAML(typ="safe").load(
        (stage.handle.root / "audio" / stage.project.audio.lyrics_new).read_text(encoding="utf-8")
    )
    if language in SOULX and voice.kind is not VoiceKind.RVC:
        if not voice.reference:
            raise ConfigError(f"voice {name} has no reference clip")
        prompt_meta = stage.rpath("audio", "voices", name, "prompt.json")
        if not stage.ex.exists(prompt_meta):
            _run(
                stage,
                "transcribe.soulx_pre",
                {"action": "preprocess", "audio": voice.reference, "language": language, "midi": False},
                stage.rpath("audio", "voices", name),
                f"voice prompt {name}",
                force,
            )
        applied = _run(
            stage,
            "svs.soulx_singer",
            {
                "action": "apply_lyrics",
                "metadata": (stage.handle.state.read("audio") or {})["metadata"],
                "lines": list(lyrics["lines"]),
                "language": language,
            },
            stage.rpath("audio", "sing"),
            "apply lyrics",
            True,
        )
        record = _run(
            stage,
            "svs.soulx_singer",
            {
                "action": "sing",
                "prompt_wav": voice.reference,
                "prompt_metadata": prompt_meta,
                "target_metadata": _first(applied, "metadata"),
                "pitch_shift": voice.pitch_shift,
            },
            stage.rpath("audio", "sing"),
            "sing",
            force,
        )
        vocal = _first(record, "vocal")
    elif voice.kind is VoiceKind.RVC:
        record = _run(
            stage,
            "svc.applio_rvc",
            {
                "action": "infer",
                "input": (stage.handle.state.read("audio") or {})["stems"]["lead_dry"],
                "model": voice.rvc_model,
                "index": voice.rvc_index,
                "pitch": voice.pitch_shift,
            },
            stage.rpath("audio", "sing"),
            "rvc",
            force,
        )
        vocal = _first(record, "vocal")
    else:
        # Japanese, Korean and other languages: repaint the mix, then convert the new lead.
        record = _run(
            stage,
            "svs.acestep_repaint",
            {
                "action": "repaint",
                "source": (stage.handle.state.read("ingest") or {})["audio"],
                "lyrics": "\n".join(lyrics["lines"]),
                "language": language,
                "caption": "the same song with new lyrics",
            },
            stage.rpath("audio", "repaint"),
            "repaint",
            force,
        )
        vocal = _first(record, "mix")
    state = {**(stage.handle.state.read("audio") or {}), "lead_new": vocal}
    stage.handle.state.write("audio", state)
    return state


def mix(stage: Stage, *, force: bool = False) -> dict[str, Any]:
    audio = stage.handle.state.read("audio") or {}
    if "lead_new" not in audio or "stems" not in audio:
        raise ConfigError("nothing to mix", hint="higgsless audio separate and higgsless audio sing")
    record = _run(
        stage,
        "mix.pedalboard_matchering",
        {
            "lead": audio["lead_new"],
            "instrumental": audio["stems"]["instrumental"],
            "reference_vocal": audio["stems"].get("lead"),
            "reference_track": (stage.handle.state.read("ingest") or {}).get("audio"),
        },
        stage.rpath("audio", "mix"),
        "mix",
        force,
    )
    state = {**audio, "final_mix": _first(record, "final_mix"), "lead_wet": _first(record, "lead_wet")}
    stage.handle.state.write("audio", state)
    return state


def set_reference_voice(stage: Stage, name: str, reference: Path, consent: Consent) -> dict[str, Any]:
    if not reference.exists():
        raise ConfigError(f"reference clip not found: {reference}")
    remote = stage.rpath("audio", "voices", name, f"reference{reference.suffix.lower()}")
    stage.ex.makedirs(stage.rpath("audio", "voices", name))
    stage.ex.put(reference, remote)
    with stage.handle.edit() as data:
        data["voices"][name] = {
            "kind": "reference",
            "consent": consent.value,
            "reference": remote,
            "language": data["audio"]["language"],
        }
        data["audio"]["lead_voice"] = name
    return {"voice": name, "reference": remote}
