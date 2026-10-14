"""Project handle: manifest loading and comment-preserving edits, state, and path helpers."""

from __future__ import annotations

import io
import ntpath
import os
import posixpath
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from higgsless.core.config import WorkstationConfig
from higgsless.core.manifest import Project
from higgsless.core.paths import projects_dir
from higgsless.core.state import StateStore
from higgsless.errors import ConfigError

MANIFEST = "project.yaml"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")

TEMPLATE = """\
# higgsless project. Edit freely; `higgsless --json validate` checks it.
version: 1
project: {name}
# private | public_noncommercial | commercial (decides which engine licences are allowed)
usage: {usage}
# original (keep sets, swap performers) | new-world (rebuild shots) | restyle (needs `style`)
look: {look}
style: null
# draft (480p, fast) | standard (720p) | max (720p, slow, best)
profile: {profile}
takes: 1
delivery:
  resolution: 1080p
  codec: h264
  crf: 18
# Filled by `higgsless init --source` / `ingest`.
source: null
# name -> {{prompt, images, style, stylised}}
characters: {{}}
# name -> {{kind: reference|designed|rvc, consent: self|written|synthetic, ...}}
voices: {{}}
# performer id (from `cast detect`) -> character name
cast: {{}}
# capability -> engine id, e.g. swap: swap.wan22_animate_mix
engines: {{}}
# engine ids allowed despite a non-commercial licence
allow_noncommercial: []
audio:
  language: en
  lead_voice: null
  backing: keep
  lyrics_new: lyrics_new.yaml
# Filled by `shots detect`; set mode/engine/cast/lipsync per shot.
shots: []
"""


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120
    return y


def remote_join(root: str, *parts: str) -> str:
    """Join remote paths, using Windows rules only when the root is a Windows path."""
    is_windows = bool(re.match(r"^[A-Za-z]:[\\/]", root)) or "\\" in root
    mod = ntpath if is_windows else posixpath
    return mod.join(root, *parts)


class ProjectHandle:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / MANIFEST
        self.state = StateStore(root / "state")
        self._project: Project | None = None

    @property
    def name(self) -> str:
        return self.load().project

    @property
    def previews_dir(self) -> Path:
        path = self.root / "previews"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, refresh: bool = False) -> Project:
        if self._project is None or refresh:
            data = _yaml().load(self.manifest_path.read_text(encoding="utf-8")) or {}
            self._project = _validate(data, self.manifest_path)
        return self._project

    @contextmanager
    def edit(self) -> Iterator[CommentedMap]:
        """Edit project.yaml in place, keeping comments; validated before it is written."""
        yaml = _yaml()
        data = yaml.load(self.manifest_path.read_text(encoding="utf-8")) or CommentedMap()
        yield data
        project = _validate(_plain(data), self.manifest_path)
        buf = io.StringIO()
        yaml.dump(data, buf)
        _atomic_write(self.manifest_path, buf.getvalue())
        self._project = project

    # Remote paths -----------------------------------------------------------------------------
    def remote_root(self, cfg: WorkstationConfig) -> str:
        return remote_join(cfg.data_root, "projects", self.name)

    def rpath(self, cfg: WorkstationConfig, *parts: str) -> str:
        return remote_join(self.remote_root(cfg), *parts)

    def input_rel(self, cfg: WorkstationConfig, *parts: str) -> str:
        """Path relative to ComfyUI's input/output directory (the data root), with forward slashes."""
        return posixpath.join("projects", self.name, *[p.replace("\\", "/") for p in parts])


def _plain(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(k): _plain(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_plain(v) for v in data]
    return data


def _validate(data: Any, where: Path) -> Project:
    try:
        return Project.model_validate(_plain(data))
    except PydanticValidationError as exc:
        errors = [{"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in exc.errors(include_url=False)]
        raise ConfigError(f"{where} is invalid", details=errors, hint="fix the listed fields") from exc


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_projects() -> list[str]:
    base = projects_dir()
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / MANIFEST).exists())


def open_project(selector: str | None) -> ProjectHandle:
    if selector:
        candidate = Path(selector)
        if candidate.name == MANIFEST and candidate.exists():
            return ProjectHandle(candidate.parent.resolve())
        if (candidate / MANIFEST).exists():
            return ProjectHandle(candidate.resolve())
        named = projects_dir() / selector
        if (named / MANIFEST).exists():
            return ProjectHandle(named)
        raise ConfigError(f"project not found: {selector}", hint=f"known projects: {list_projects() or 'none'}")
    names = list_projects()
    if len(names) == 1:
        return ProjectHandle(projects_dir() / names[0])
    if not names:
        raise ConfigError("no projects yet", hint="run `higgsless init <name> --source <video>`")
    raise ConfigError("several projects exist; pass --project", hint=f"known projects: {names}")


def create_project(name: str, *, usage: str, look: str, profile: str, exist_ok: bool = False) -> ProjectHandle:
    if not _NAME.match(name):
        raise ConfigError("project names use lowercase letters, digits, '-' and '_' (2-63 chars)")
    root = projects_dir() / name
    manifest = root / MANIFEST
    if manifest.exists():
        if exist_ok:
            return ProjectHandle(root)
        raise ConfigError(f"project {name} already exists at {root}")
    root.mkdir(parents=True, exist_ok=True)
    text = TEMPLATE.format(name=name, usage=usage, look=look, profile=profile)
    _validate(_yaml().load(text), manifest)
    _atomic_write(manifest, text)
    (root / "state").mkdir(exist_ok=True)
    return ProjectHandle(root)
