"""Build a contact sheet on the workstation from images already there."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.qa.contact_sheet import contact_sheet
from higgsless.worker_tasks._harness import run_task


def task(params: dict[str, Any], out: Path) -> dict[str, Any]:
    items = [(item["path"], item["label"]) for item in params["items"] if Path(item["path"]).exists()]
    dst = out / params.get("name", "contact.jpg")
    contact_sheet(
        items,
        dst,
        columns=params.get("columns"),
        cell_width=int(params.get("cell_width", 320)),
        title=params.get("title"),
    )
    return {"outputs": {"sheet": [str(dst)]}}


if __name__ == "__main__":
    run_task(task)
