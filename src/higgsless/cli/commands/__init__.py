"""Command registration."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    from higgsless.cli.commands import (
        bench_cmd,
        config_cmd,
        engines_cmd,
        models_cmd,
        worker_cmd,
        workflows_cmd,
    )

    app.add_typer(config_cmd.app, name="config")
    app.add_typer(engines_cmd.app, name="engines")
    app.add_typer(workflows_cmd.app, name="workflows")
    app.add_typer(models_cmd.app, name="models")
    app.add_typer(worker_cmd.app, name="worker")
    app.command("bench")(bench_cmd.bench)
