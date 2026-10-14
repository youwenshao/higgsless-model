"""JSON encoding shared by the CLI output and the state store."""

from __future__ import annotations

import enum
import json
from fractions import Fraction
from pathlib import PurePath
from typing import Any


def _default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def to_json(data: Any, indent: int | None = 2) -> str:
    return json.dumps(data, default=_default, ensure_ascii=False, indent=indent)
