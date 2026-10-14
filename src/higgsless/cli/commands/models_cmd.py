"""`higgsless models ...`: model files required by engines, and syncing them to the workstation."""

from __future__ import annotations

from typing import Any

import typer

from higgsless.cli.context import AppContext, get_ctx
from higgsless.cli.output import table
from higgsless.core.models import ModelsLock, check_remote, models_for, sync_models
from higgsless.engines.registry import registry
from higgsless.errors import ConfigError

app = typer.Typer(help="Model files per engine (worker/models.lock.yaml).", no_args_is_help=True)


def _selected_engines(app_ctx: AppContext, engine_ids: list[str], all_engines: bool) -> list[Any]:
    reg = registry()
    if engine_ids:
        return [reg.get(i) for i in engine_ids]
    if all_engines:
        return list(reg.engines.values())
    project = None
    try:
        project = app_ctx.project.load()
    except ConfigError:
        project = None
    capabilities = sorted({e.capability for e in reg.engines.values()})
    chosen = {reg.default_for(cap, project).id for cap in capabilities}
    return [reg.get(i) for i in sorted(chosen)]


def _model_ids(engines: list[Any], profile: str) -> list[str]:
    ids: list[str] = []
    for engine in engines:
        for model_id in models_for(engine.models, profile):
            if model_id not in ids:
                ids.append(model_id)
    return ids


@app.command("list")
def list_models(
    ctx: typer.Context,
    profile: str = typer.Option("standard", help="draft | standard | max"),
    engine_ids: list[str] = typer.Option([], "--engine", "-e"),
    all_engines: bool = typer.Option(False, "--all", help="Every engine, not just the defaults."),
    check: bool = typer.Option(False, help="Check which files exist on the workstation."),
) -> None:
    """List model files the selected engines need."""
    app_ctx = get_ctx(ctx)
    lock = ModelsLock.load()
    ids = _model_ids(_selected_engines(app_ctx, engine_ids, all_engines), profile)
    data = []
    for model_id in ids:
        model = lock.get(model_id)
        row: dict[str, Any] = {
            "id": model_id,
            "repo": model.repo,
            "file": model.file,
            "dest": model.dest,
            "size_gb": round((model.size or 0) / 1e9, 2),
            "licence": model.licence,
        }
        if check:
            row.update(check_remote(app_ctx.executor, app_ctx.config, model))
        data.append(row)
    total = round(sum(d["size_gb"] for d in data), 1)
    rows = [[d["id"], d["dest"], d["size_gb"], d.get("status", "")] for d in data]
    app_ctx.out.result(
        {"profile": profile, "total_gb": total, "models": data},
        table(f"models ({profile}, {total} GB)", ["id", "dest", "GB", "status"], rows),
    )


@app.command("sync")
def sync(
    ctx: typer.Context,
    profile: str = typer.Option("standard", help="draft | standard | max"),
    engine_ids: list[str] = typer.Option([], "--engine", "-e"),
    all_engines: bool = typer.Option(False, "--all"),
    verify: bool = typer.Option(False, help="Verify sha256 of every file (slow)."),
    dry_run: bool = typer.Option(False, help="Only report what would be downloaded."),
) -> None:
    """Download missing model files on the workstation (resumable)."""
    app_ctx = get_ctx(ctx)
    lock = ModelsLock.load()
    ids = _model_ids(_selected_engines(app_ctx, engine_ids, all_engines), profile)
    if dry_run:
        status = [{"id": i, **check_remote(app_ctx.executor, app_ctx.config, lock.get(i))} for i in ids]
        app_ctx.out.result({"profile": profile, "models": status})
        return
    results = sync_models(app_ctx.executor, app_ctx.config, lock, ids, verify=verify, progress=app_ctx.out.progress)
    app_ctx.out.result({"profile": profile, "models": results}, f"{len(results)} model file(s) present")
