"""Locate repo data folders (engines/, workflows/, worker/) and the projects folder."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

_MARKERS = ("engines", "workflows", "worker")


@cache
def repo_root() -> Path | None:
    env = os.environ.get("HIGGSLESS_HOME")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [*here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if all((parent / m).is_dir() for m in _MARKERS) and (parent / "pyproject.toml").exists():
            return parent
    return None


def data_dir(name: str) -> Path:
    """engines/, workflows/ or worker/: from the repo checkout, else the copy bundled in the wheel."""
    root = repo_root()
    if root is not None and (root / name).is_dir():
        return root / name
    bundled = Path(__file__).resolve().parent.parent / "_data" / name
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(f"cannot find the {name}/ folder; set HIGGSLESS_HOME to the repo checkout")


def projects_dir() -> Path:
    env = os.environ.get("HIGGSLESS_PROJECTS")
    if env:
        return Path(env).resolve()
    root = repo_root()
    return (root / "projects") if root else (Path.cwd() / "projects")


def config_dir() -> Path:
    root = repo_root()
    return (root / "config") if root else (Path.home() / ".config" / "higgsless")
