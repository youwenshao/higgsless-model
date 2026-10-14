"""Run the GPL mixer as a separate process, so the higgsless package never imports it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from higgsless.worker_tasks._harness import run_task

_SCRIPT = Path(os.environ.get("HIGGSLESS_WORKER", "/opt/higgsless")) / "worker" / "gpl" / "mix.py"


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    params_path = out / "_mix_params.json"
    import json

    params_path.write_text(json.dumps(params), encoding="utf-8")
    result_path = out / "_mix_result.json"
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--params-file",
            str(params_path),
            "--output-dir",
            str(out),
            "--result",
            str(result_path),
        ],
        check=True,
    )
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not data.get("ok", True):
        raise RuntimeError(data.get("error"))
    return data


if __name__ == "__main__":
    run_task(task)
