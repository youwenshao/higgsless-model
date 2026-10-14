"""`higgsless workflows ...`: validate and render ComfyUI workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.engines.bindings import bind
from higgsless.engines.registry import registry
from higgsless.engines.runner import load_workflow
from higgsless.engines.validate import validate_workflow
from higgsless.errors import ValidationError
from higgsless.remote.comfy import ComfyClient

app = typer.Typer(help="Validate and inspect ComfyUI workflows.", no_args_is_help=True)

_PLACEHOLDER = {
    "input_path": "placeholder.png",
    "abs_path": "/placeholder",
    "int": 1,
    "float": 1.0,
    "bool": False,
    "seed": 1,
    "str": "placeholder",
    "json": {},
}


def _object_info(app_ctx: Any, comfy_url: str | None, object_info: Path | None, save: Path | None) -> dict:
    if object_info is not None:
        return json.loads(object_info.read_text(encoding="utf-8"))
    client = ComfyClient(comfy_url) if comfy_url else app_ctx.comfy
    info = client.object_info()
    if save is not None:
        save.write_text(json.dumps(info), encoding="utf-8")
    return info


def _dummy_params(engine: Any, bindings: Any) -> dict[str, Any]:
    params = engine.params_for("standard")
    params = {k: v for k, v in params.items() if k in bindings.params}
    for name, b in bindings.params.items():
        if name not in params and b.required and b.default is None:
            params[name] = _PLACEHOLDER[b.type]
    return params


@app.command("validate")
def validate(
    ctx: typer.Context,
    engine_ids: list[str] = typer.Option([], "--engine", "-e", help="Only these engines."),
    comfy_url: str | None = typer.Option(None, help="ComfyUI URL (default from config)."),
    object_info: Path | None = typer.Option(None, help="Validate against a saved /object_info JSON."),
    save_object_info: Path | None = typer.Option(None, help="Save the fetched /object_info here."),
    strict: bool = typer.Option(False, help="Treat unknown combo values (e.g. missing model files) as errors."),
) -> None:
    """Check every ComfyUI engine's workflow against the server's node definitions."""
    app_ctx = get_ctx(ctx)
    info = _object_info(app_ctx, comfy_url, object_info, save_object_info)
    engines = [e for e in registry().engines.values() if e.backend == "comfyui"]
    if engine_ids:
        engines = [registry().get(i) for i in engine_ids]
    report: dict[str, Any] = {}
    errors = 0
    for engine in sorted(engines, key=lambda e: e.id):
        workflow, bindings = load_workflow(engine)
        issues = validate_workflow(workflow, info, strict_combos=strict)
        bound = bind(workflow, bindings, _dummy_params(engine, bindings))
        bound_issues = [i for i in validate_workflow(bound.workflow, info, strict_combos=strict) if i.level == "error"]
        seen = {(i.node, i.message) for i in issues}
        issues += [i for i in bound_issues if (i.node, i.message) not in seen]
        report[engine.id] = {
            "errors": [i.to_dict() for i in issues if i.level == "error"],
            "warnings": [i.to_dict() for i in issues if i.level == "warning"],
            "bypassed_in_minimal_variant": bound.bypassed,
        }
        errors += len(report[engine.id]["errors"])
    rows = [[k, len(v["errors"]), len(v["warnings"])] for k, v in report.items()]
    if errors:
        raise ValidationError(f"{errors} workflow error(s)", details=report)
    app_ctx.out.result(report, table("workflow validation", ["engine", "errors", "warnings"], rows))


@app.command("render")
def render(
    ctx: typer.Context,
    engine_id: str,
    param: list[str] = typer.Option([], "--param", "-P", help="key=value (JSON values allowed)."),
    profile: str = typer.Option("standard", help="Profile whose defaults apply."),
) -> None:
    """Print the bound API workflow for an engine (useful for debugging in the ComfyUI editor)."""
    app_ctx = get_ctx(ctx)
    engine = registry().get(engine_id)
    workflow, bindings = load_workflow(engine)
    overrides: dict[str, Any] = {}
    for item in param:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    params = {k: v for k, v in engine.params_for(profile, overrides).items() if k in bindings.params}
    bound = bind(workflow, bindings, params)
    app_ctx.out.result({"workflow": bound.workflow, "outputs": bound.outputs, "bypassed": bound.bypassed})
