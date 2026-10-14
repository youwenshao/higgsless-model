"""Applio (MIT): convert with a trained RVC voice, or train one."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import log, run_task

APPLIO = os.environ.get("APPLIO_REPO", "/opt/applio")


def _run(args: list[str]) -> None:
    log(" ".join(args))
    subprocess.run(args, cwd=APPLIO, check=True)


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    action = params["action"]
    if action == "infer":
        dst = out / "converted.wav"
        _run(
            [
                "python",
                "core.py",
                "infer",
                "--input-path",
                params["input"],
                "--output-path",
                str(dst),
                "--pth-path",
                params["model"],
                "--index-path",
                params["index"],
                "--pitch",
                str(params.get("pitch", 0)),
                "--f0-method",
                params.get("f0_method", "rmvpe"),
                "--index-rate",
                str(params.get("index_rate", 0.5)),
                "--protect",
                str(params.get("protect", 0.4)),
                "--export-format",
                "wav",
            ]
        )
        return {"outputs": {"vocal": [str(dst)]}}
    if action == "train":
        name = params["name"]
        common = ["--model-name", name, "--sample-rate", str(params.get("sample_rate", 48000))]
        _run(["python", "core.py", "preprocess", *common, "--dataset-path", params["dataset"]])
        _run(["python", "core.py", "extract", *common, "--f0-method", params.get("f0_method", "rmvpe"), "--gpu", "0"])
        _run(
            [
                "python",
                "core.py",
                "train",
                *common,
                "--total-epoch",
                str(params.get("total_epoch", 300)),
                "--batch-size",
                str(params.get("batch_size", 8)),
                "--gpu",
                "0",
                "--save-every-epoch",
                "50",
            ]
        )
        _run(["python", "core.py", "index", "--model-name", name])
        logs = Path(APPLIO) / "logs" / name
        weights = sorted(logs.glob("*.pth")) + sorted((Path(APPLIO) / "logs").rglob(f"{name}*.pth"))
        indexes = sorted(logs.rglob("*.index"))
        if not weights or not indexes:
            raise RuntimeError(f"training finished but no .pth/.index under {logs}")
        return {"outputs": {"model": [str(weights[-1])], "index": [str(indexes[-1])]}}
    raise ValueError(f"unknown action {action!r}")


if __name__ == "__main__":
    run_task(task)
