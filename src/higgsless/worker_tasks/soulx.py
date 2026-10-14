"""SoulX-Singer inside its container: preprocess, score-mode singing, and SVC."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import log, run_task

REPO = os.environ.get("SOULX_REPO", "/opt/soulx")
LANG = {
    "en": "English",
    "zh": "Mandarin",
    "yue": "Cantonese",
    "english": "English",
    "mandarin": "Mandarin",
    "cantonese": "Cantonese",
}


def _models(root: Path) -> Path:
    """Snapshots into a writable dir; preprocess and inference use relative pretrained_models/ paths."""
    from huggingface_hub import snapshot_download

    base = root / "pretrained_models"
    base.mkdir(parents=True, exist_ok=True)
    for repo, name in (
        ("Soul-AILab/SoulX-Singer", "SoulX-Singer"),
        ("Soul-AILab/SoulX-Singer-Preprocess", "SoulX-Singer-Preprocess"),
    ):
        link = base / name
        if not link.exists():
            snap = Path(snapshot_download(repo))
            link.symlink_to(snap, target_is_directory=True)
            log(f"{name} -> {snap}")
    return base


def _run(args: list[str], cwd: Path) -> None:
    env = {**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
    log(" ".join(args))
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _apply_lyrics(params: dict[str, Any], out: Path) -> dict[str, Any]:
    """Lay new lyric lines onto the transcribed notes and rebuild phonemes. Timing stays put."""
    import json
    import sys

    sys.path.insert(0, REPO)
    from preprocess.tools.g2p import g2p_transform

    from higgsless.audio.syllables import units

    language = params.get("language", "en").lower()
    lang = LANG.get(language, "English")
    data = json.loads(Path(params["metadata"]).read_text(encoding="utf-8"))
    lines = params["lines"]
    if len(lines) != len(data):
        raise RuntimeError(f"{len(lines)} lyric lines for {len(data)} segments")
    for segment, line in zip(data, lines, strict=True):
        pitches = str(segment.get("note_pitch", "")).split()
        sung = [p for p in pitches if p not in {"0", "0.0"}]
        pieces = units(line, language)
        if len(pieces) != len(sung):
            raise RuntimeError(
                f"{segment.get('index')}: {len(pieces)} words for {len(sung)} sung notes ({segment.get('text')!r})"
            )
        pending = iter(pieces)
        tokens = ["<SP>" if pitch in {"0", "0.0"} else next(pending) for pitch in pitches]
        segment["text"] = " ".join(tokens)
        segment["phoneme"] = " ".join(str(p) for p in g2p_transform(tokens, lang))
    dst = out / "metadata.json"
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"outputs": {"metadata": [str(dst)]}}


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    action = params["action"]
    if action == "apply_lyrics":
        return _apply_lyrics(params, out)
    work = Path(params.get("model_dir") or out / "_models")
    _models(work)
    if action == "preprocess":
        save = out / "transcription"
        save.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "python",
                "-m",
                "preprocess.pipeline",
                "--audio_path",
                params["audio"],
                "--save_dir",
                str(save),
                "--language",
                LANG.get(params.get("language", "en").lower(), params.get("language", "English")),
                "--device",
                params.get("device", "cuda"),
                "--vocal_sep",
                "False",
                "--midi_transcribe",
                "True" if params.get("midi", True) else "False",
            ],
            work,
        )
        meta = save / "metadata.json"
        return {"outputs": {"metadata": [str(meta)], "midi_dir": [str(save)]}}

    if action == "sing":
        _run(
            [
                "python",
                "-m",
                "cli.inference",
                "--device",
                params.get("device", "cuda"),
                "--model_path",
                str(work / "pretrained_models/SoulX-Singer/model.pt"),
                "--config",
                f"{REPO}/soulxsinger/config/soulxsinger.yaml",
                "--prompt_wav_path",
                params["prompt_wav"],
                "--prompt_metadata_path",
                params["prompt_metadata"],
                "--target_metadata_path",
                params["target_metadata"],
                "--phoneset_path",
                f"{REPO}/soulxsinger/utils/phoneme/phone_set.json",
                "--save_dir",
                str(out),
                "--control",
                params.get("control", "score"),
                "--pitch_shift",
                str(params.get("pitch_shift", 0)),
                *(["--auto_shift"] if params.get("auto_shift", True) else []),
                "--fp16",
            ],
            work,
        )
        wav = out / "generated.wav"
        if not wav.exists():
            found = sorted(out.glob("*.wav"))
            if not found:
                raise RuntimeError("SoulX-Singer wrote no wav")
            wav = found[-1]
        return {"outputs": {"vocal": [str(wav)]}}

    if action == "svc":
        _run(
            [
                "python",
                "-m",
                "cli.inference_svc",
                "--device",
                params.get("device", "cuda"),
                "--model_path",
                str(work / "pretrained_models/SoulX-Singer/model-svc.pt"),
                "--config",
                f"{REPO}/soulxsinger/config/soulxsinger.yaml",
                "--prompt_wav_path",
                params["prompt_wav"],
                "--target_wav_path",
                params["target_wav"],
                "--prompt_f0_path",
                params["prompt_f0"],
                "--target_f0_path",
                params["target_f0"],
                "--save_dir",
                str(out),
                "--pitch_shift",
                str(params.get("pitch_shift", 0)),
                *(["--auto_shift"] if params.get("auto_shift", True) else []),
                "--fp16",
            ],
            work,
        )
        wavs = sorted(out.glob("*.wav"))
        if not wavs:
            raise RuntimeError("SoulX-Singer-SVC wrote no wav")
        return {"outputs": {"vocal": [str(wavs[-1])]}}
    raise ValueError(f"unknown action {action!r}")


if __name__ == "__main__":
    run_task(task)
