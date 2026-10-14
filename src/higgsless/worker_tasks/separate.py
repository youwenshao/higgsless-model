"""RoFormer separation: vocals, then lead vs backing, then a dry lead and a reverb stem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import log, run_task


def _separate(src: str, model: str, out: Path, name: str) -> list[str]:
    from audio_separator.separator import Separator

    before = {p.name for p in out.glob("*.wav")}
    separator = Separator(output_dir=str(out), output_format="wav")
    separator.load_model(model_filename=model)
    produced = separator.separate(src)
    paths = []
    for item in produced:
        path = Path(item)
        if not path.is_absolute():
            path = out / path.name
        paths.append(str(path))
    if not paths:  # some versions only write files and return names we already have
        paths = [str(p) for p in out.glob("*.wav") if p.name not in before]
    log(f"{name}: {len(paths)} stem(s) with {model}")
    return paths


def _pick(paths: list[str], *needles: str) -> str:
    for needle in needles:
        for path in paths:
            if needle.lower() in Path(path).name.lower():
                return path
    raise RuntimeError(f"no stem matching {needles} in {[Path(p).name for p in paths]}")


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    src = params["source"]
    vocals = _separate(src, params["vocals_model"], out / "vocals", "vocals")
    vocal = _pick(vocals, "vocal")
    instrumental = _pick(vocals, "instrumental", "no_vocals", "other")
    karaoke = _separate(vocal, params["karaoke_model"], out / "karaoke", "karaoke")
    lead = _pick(karaoke, "vocal", "lead")
    backing = _pick(karaoke, "instrumental", "backing", "other")
    dry = _separate(lead, params["dereverb_model"], out / "dereverb", "dereverb")
    lead_dry = _pick(dry, "noreverb", "dry", "no_reverb")
    try:
        reverb = _pick(dry, "reverb")
    except RuntimeError:
        reverb = None
    outputs = {
        "instrumental": [instrumental],
        "vocals": [vocal],
        "lead": [lead],
        "backing": [backing],
        "lead_dry": [lead_dry],
    }
    if reverb and reverb != lead_dry:
        outputs["reverb"] = [reverb]
    return {"outputs": outputs}


if __name__ == "__main__":
    run_task(task)
