"""Content hashing for job caching."""

from __future__ import annotations

import enum
import hashlib
import json
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any


def _default(obj: Any) -> Any:
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"cannot hash {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def hash_obj(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
