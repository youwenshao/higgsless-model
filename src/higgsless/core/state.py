"""Machine-written project state: small JSON documents under projects/<name>/state/."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from higgsless.core.jsonio import to_json


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        if not key or key.startswith(("/", "\\")) or ".." in Path(key).parts:
            raise ValueError(f"bad state key {key!r}")
        return self.root / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, key: str, data: Any) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(to_json(data) + "\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        data = fn(self.read(key, default))
        self.write(key, data)
        return data

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def keys(self, prefix: str = "") -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        keys = []
        for path in sorted(base.rglob("*.json")):
            rel = path.relative_to(self.root).with_suffix("")
            keys.append(rel.as_posix())
        return keys
