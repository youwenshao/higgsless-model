"""Command registration."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    from higgsless.cli.commands import (
        analysis_cmd,
        bench_cmd,
        config_cmd,
        doctor_cmd,
        engines_cmd,
        models_cmd,
        project_cmd,
        shots_cmd,
        worker_cmd,
        workflows_cmd,
    )

    app.command("init")(project_cmd.init)
    app.command("ingest")(project_cmd.ingest)
    app.command("validate")(project_cmd.validate)
    app.add_typer(shots_cmd.app, name="shots")
    app.command("analyze")(analysis_cmd.analyze)
    app.add_typer(analysis_cmd.cast_app, name="cast")
    app.command("doctor")(doctor_cmd.doctor)
    app.add_typer(config_cmd.app, name="config")
    app.add_typer(engines_cmd.app, name="engines")
    app.add_typer(workflows_cmd.app, name="workflows")
    app.add_typer(models_cmd.app, name="models")
    app.add_typer(worker_cmd.app, name="worker")
    app.command("bench")(bench_cmd.bench)
