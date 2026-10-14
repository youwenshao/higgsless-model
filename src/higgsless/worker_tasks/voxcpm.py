"""VoxCPM2: design an invented voice from a text description and render a reference clip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import run_task

SAMPLE = (
    "I am singing just for you tonight, holding every note as long as the melody asks me to, "
    "softly at first and then with the whole chorus behind me."
)


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    import soundfile as sf
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    text = params.get("text") or SAMPLE
    wav = model.generate(
        text=f"({params['describe']}){text}",
        cfg_value=float(params.get("cfg_value", 2.0)),
        inference_timesteps=int(params.get("steps", 10)),
    )
    dst = out / "reference.wav"
    sf.write(str(dst), wav, model.tts_model.sample_rate)
    return {"outputs": {"reference": [str(dst)]}, "sample_rate": model.tts_model.sample_rate}


if __name__ == "__main__":
    run_task(task)
