"""ACE-Step 1.5 repaint: regenerate a section (or the whole song) with new lyrics, keeping the track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import log, run_task


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    root = params.get("project_root", "/opt/acestep")
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=root, config_path=params.get("config", "acestep-v15-turbo"), device=params.get("device", "cuda")
    )
    log(f"dit: {status} ({ok})")
    if not ok:
        raise RuntimeError(f"ACE-Step init failed: {status}")
    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir=params.get("checkpoints", root),
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend=params.get("backend", "vllm"),
        device=params.get("device", "cuda"),
    )
    gen = GenerationParams(
        task_type="repaint",
        src_audio=params["source"],
        caption=params.get("caption", "the same song"),
        lyrics=params.get("lyrics", ""),
        vocal_language=params.get("language", "unknown"),
        repainting_start=float(params.get("start", 0.0)),
        repainting_end=float(params.get("end", -1)),
    )
    result = generate_music(dit, llm, gen, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=str(out))
    if not getattr(result, "success", False):
        raise RuntimeError(f"ACE-Step repaint failed: {getattr(result, 'error', result)}")
    paths = [a["path"] for a in result.audios]
    return {"outputs": {"mix": paths}}


if __name__ == "__main__":
    run_task(task)
