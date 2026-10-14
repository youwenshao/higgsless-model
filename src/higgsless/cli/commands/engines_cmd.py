"""`higgsless engines ...`: inspect the engine registry."""

from __future__ import annotations

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.core.paths import data_dir
from higgsless.engines.bindings import Bindings
from higgsless.engines.registry import registry

app = typer.Typer(help="Inspect engines (models wired into workflows or tools).", no_args_is_help=True)


@app.command("list")
def list_engines(
    ctx: typer.Context,
    capability: str | None = typer.Option(None, "--capability", "-c", help="Filter by capability."),
) -> None:
    """List engines with capability, backend, licence and default flag."""
    app_ctx = get_ctx(ctx)
    engines = sorted(registry().engines.values(), key=lambda e: (e.capability, not e.default, e.id))
    if capability:
        engines = [e for e in engines if e.capability == capability]
    data = [
        {
            "id": e.id,
            "capability": e.capability,
            "backend": e.backend,
            "default": e.default,
            "experimental": e.experimental,
            "licence": e.licence.weights,
            "commercial": e.licence.commercial,
            "title": e.title,
        }
        for e in engines
    ]
    rows = [
        [
            d["id"],
            d["capability"],
            d["backend"],
            "yes" if d["default"] else "",
            d["licence"],
            "yes" if d["commercial"] else "no",
        ]
        for d in data
    ]
    app_ctx.out.result(
        data, table("engines", ["id", "capability", "backend", "default", "licence", "commercial"], rows)
    )


@app.command("show")
def show(ctx: typer.Context, engine_id: str) -> None:
    """Show one engine manifest, including its workflow parameters."""
    app_ctx = get_ctx(ctx)
    engine = registry().get(engine_id)
    data = engine.model_dump(mode="json")
    if engine.bindings:
        bindings = Bindings.load(data_dir("workflows") / engine.bindings)
        data["parameters"] = {
            name: {
                "type": b.type,
                "required": b.required,
                "default": b.default,
                "optional": bool(b.optional_nodes),
                "description": b.description,
            }
            for name, b in bindings.params.items()
        }
        data["outputs"] = list(bindings.outputs)
    app_ctx.out.result(data)
