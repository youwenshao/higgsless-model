"""Check API-format workflows against a ComfyUI server's /object_info node definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Inputs that accept any type.
_WILDCARDS = {"*", "ANY", "any"}


@dataclass
class Issue:
    level: str  # error | warning
    node: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _input_specs(info: dict[str, Any]) -> dict[str, Any]:
    spec = info.get("input") or {}
    merged: dict[str, Any] = {}
    for section in ("required", "optional", "hidden"):
        merged.update(spec.get(section) or {})
    return merged


def _required(info: dict[str, Any]) -> set[str]:
    return set(((info.get("input") or {}).get("required") or {}).keys())


def _input_type(spec: Any) -> tuple[str | None, list[Any] | None, dict[str, Any]]:
    """Return (type name, combo options, options dict) for an input spec."""
    if not isinstance(spec, list | tuple) or not spec:
        return None, None, {}
    head = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(head, list):
        return "COMBO", head, opts
    if head == "COMBO":
        return "COMBO", list(opts.get("options") or []), opts
    return str(head), None, opts


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _autogrow_owner(name: str, specs: dict[str, Any]) -> str | None:
    """Dynamic inputs like `reference_images.reference_image_0` belong to a parent input."""
    if "." in name:
        parent = name.split(".", 1)[0]
        if parent in specs:
            return parent
    return None


def validate_workflow(
    workflow: dict[str, Any], object_info: dict[str, Any], *, strict_combos: bool = False
) -> list[Issue]:
    issues: list[Issue] = []
    for node_id, node in workflow.items():
        cls = node.get("class_type")
        title = (node.get("_meta") or {}).get("title", "")
        label = f"{node_id} ({title or cls})"
        info = object_info.get(cls)
        if info is None:
            issues.append(Issue("error", label, f"unknown node class {cls!r} (missing node pack?)"))
            continue
        specs = _input_specs(info)
        inputs = node.get("inputs", {})
        for name, value in inputs.items():
            spec = specs.get(name)
            if spec is None:
                if _autogrow_owner(name, specs):
                    continue
                issues.append(Issue("error", label, f"unknown input {name!r} for {cls}"))
                continue
            kind, options, opts = _input_type(spec)
            if _is_link(value):
                src = workflow.get(value[0])
                if src is None:
                    issues.append(Issue("error", label, f"input {name!r} links to missing node {value[0]}"))
                    continue
                src_info = object_info.get(src.get("class_type"))
                if src_info is None:
                    continue
                outputs = src_info.get("output") or []
                if value[1] >= len(outputs):
                    issues.append(
                        Issue(
                            "error", label, f"input {name!r} links to output {value[1]} of a node with {len(outputs)}"
                        )
                    )
                    continue
                out_type = outputs[value[1]]
                if kind and kind not in _WILDCARDS and out_type not in _WILDCARDS and kind != "COMBO":
                    accepted = {t.strip() for t in str(kind).split(",")}
                    if out_type not in accepted and not ({str(out_type)} & accepted):
                        issues.append(Issue("error", label, f"input {name!r} expects {kind}, linked to {out_type}"))
                continue
            if kind == "COMBO" and options is not None and value not in options:
                level = "error" if strict_combos else "warning"
                issues.append(Issue(level, label, f"{name}={value!r} is not among the server's options"))
            elif kind in {"INT", "FLOAT"}:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    issues.append(Issue("error", label, f"{name} expects {kind}, got {value!r}"))
                else:
                    lo, hi = opts.get("min"), opts.get("max")
                    if (lo is not None and value < lo) or (hi is not None and value > hi):
                        issues.append(Issue("error", label, f"{name}={value} outside [{lo}, {hi}]"))
        for name in _required(info) - set(inputs):
            if not any(k.startswith(name + ".") for k in inputs):
                issues.append(Issue("error", label, f"missing required input {name!r}"))
    return issues
