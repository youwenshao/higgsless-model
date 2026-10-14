"""Fill a ComfyUI API-format workflow from logical parameters.

bindings.yaml format::

    params:
      reference_image: {target: "Reference Image.image", type: input_path}
      seed: {target: ["Sampler High.noise_seed", "Sampler Low.noise_seed"], type: seed}
      speed_lora: {target: "Speed LoRA.lora_name", type: str, optional_node: "Speed LoRA"}
    bypass:
      "Speed LoRA": {0: model}     # when bypassed, consumers of output 0 get the node's `model` input
      "Continue Frames": remove     # when bypassed, consumers drop the input (it must be optional)
    outputs:
      frames: {node: "Save Frames", prefix_input: filename_prefix}

Targets address nodes by `_meta.title` (or `#<id>`), so bindings survive re-exports.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from higgsless.errors import ValidationError

PARAM_TYPES = {"str", "int", "float", "bool", "seed", "input_path", "abs_path", "json"}


class ParamBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | list[str]
    type: str = "str"
    required: bool = False
    default: Any = None
    optional_node: str | list[str] | None = None
    description: str | None = None

    @property
    def targets(self) -> list[str]:
        return [self.target] if isinstance(self.target, str) else list(self.target)

    @property
    def optional_nodes(self) -> list[str]:
        if self.optional_node is None:
            return []
        return [self.optional_node] if isinstance(self.optional_node, str) else list(self.optional_node)


class OutputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    prefix_input: str | None = "filename_prefix"
    kind: str = "images"


class Bindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, ParamBinding] = Field(default_factory=dict)
    bypass: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, OutputBinding] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Bindings:
        raw = YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {}
        bindings = cls.model_validate(raw)
        for name, p in bindings.params.items():
            if p.type not in PARAM_TYPES:
                raise ValidationError(f"{path.name}: param {name} has unknown type {p.type!r}")
        return bindings


@dataclass
class BoundWorkflow:
    workflow: dict[str, Any]
    outputs: dict[str, str]  # logical output -> node id
    params: dict[str, Any] = field(default_factory=dict)
    bypassed: list[str] = field(default_factory=list)


def _index_titles(workflow: dict[str, Any]) -> dict[str, str]:
    titles: dict[str, str] = {}
    duplicates: set[str] = set()
    for node_id, node in workflow.items():
        title = (node.get("_meta") or {}).get("title")
        if not title:
            continue
        if title in titles:
            duplicates.add(title)
        titles[title] = node_id
    for title in duplicates:
        titles.pop(title, None)
    return titles


def _resolve_node(ref: str, workflow: dict[str, Any], titles: dict[str, str]) -> str:
    if ref.startswith("#"):
        node_id = ref[1:]
        if node_id in workflow:
            return node_id
    elif ref in titles:
        return titles[ref]
    raise ValidationError(f"binding refers to missing or duplicated node {ref!r}")


def _split_target(target: str) -> tuple[str, str]:
    node, sep, inp = target.rpartition(".")
    if not sep or not node or not inp:
        raise ValidationError(f"binding target must be '<node title>.<input>', got {target!r}")
    return node, inp


def _coerce(name: str, kind: str, value: Any) -> Any:
    try:
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if kind == "seed":
            return random.randint(0, 2**48) if value in (None, "random", -1) else int(value)
        if kind in {"str", "input_path", "abs_path"}:
            return str(value).replace("\\", "/") if kind == "input_path" else str(value)
        return value
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"parameter {name} expects {kind}, got {value!r}") from exc


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _bypass(workflow: dict[str, Any], node_id: str, rule: Any) -> None:
    node = workflow.pop(node_id)
    for other in workflow.values():
        inputs = other.get("inputs", {})
        for name, value in list(inputs.items()):
            if not (_is_link(value) and value[0] == node_id):
                continue
            if rule == "remove" or rule is None:
                del inputs[name]
                continue
            slot_map = {int(k): v for k, v in dict(rule).items()}
            source_input = slot_map.get(value[1])
            if source_input is None or source_input not in node.get("inputs", {}):
                raise ValidationError(f"cannot bypass node {node_id}: no passthrough for output {value[1]}")
            inputs[name] = copy.deepcopy(node["inputs"][source_input])


def bind(
    workflow: dict[str, Any],
    bindings: Bindings,
    params: dict[str, Any],
    output_prefixes: dict[str, str] | None = None,
    *,
    allow_extra: bool = False,
) -> BoundWorkflow:
    wf = copy.deepcopy(workflow)
    titles = _index_titles(wf)

    unknown = set(params) - set(bindings.params)
    if unknown and not allow_extra:
        raise ValidationError(
            f"unknown workflow parameters: {sorted(unknown)}", hint=f"known: {sorted(bindings.params)}"
        )

    resolved: dict[str, Any] = {}
    to_bypass: list[str] = []
    for name, binding in bindings.params.items():
        value = params.get(name, binding.default)
        if value is None:
            if binding.optional_nodes:
                to_bypass.extend(binding.optional_nodes)
                continue
            if binding.required:
                raise ValidationError(f"missing required workflow parameter {name!r}")
            continue
        value = _coerce(name, binding.type, value)
        resolved[name] = value
        for target in binding.targets:
            node_ref, input_name = _split_target(target)
            node_id = _resolve_node(node_ref, wf, titles)
            wf[node_id].setdefault("inputs", {})[input_name] = value

    bypassed: list[str] = []
    for ref in dict.fromkeys(to_bypass):
        node_id = _resolve_node(ref, wf, titles)
        _bypass(wf, node_id, bindings.bypass.get(ref, "remove"))
        bypassed.append(ref)
        titles = _index_titles(wf)

    outputs: dict[str, str] = {}
    for name, out in bindings.outputs.items():
        node_id = _resolve_node(out.node, wf, titles)
        outputs[name] = node_id
        prefix = (output_prefixes or {}).get(name)
        if prefix is not None and out.prefix_input:
            wf[node_id]["inputs"][out.prefix_input] = prefix.replace("\\", "/")

    for node_id, node in wf.items():
        for input_name, value in node.get("inputs", {}).items():
            if _is_link(value) and value[0] not in wf:
                raise ValidationError(f"node {node_id}.{input_name} links to removed node {value[0]}")
    return BoundWorkflow(workflow=wf, outputs=outputs, params=resolved, bypassed=bypassed)
