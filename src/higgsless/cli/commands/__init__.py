"""Command registration."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    from higgsless.cli.commands import (
        analysis_cmd,
        audio_cmd,
        bench_cmd,
        config_cmd,
        doctor_cmd,
        engines_cmd,
        images_cmd,
        models_cmd,
        project_cmd,
        render_cmd,
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
    app.add_typer(images_cmd.char_app, name="char")
    app.add_typer(images_cmd.keyframe_app, name="keyframe")
    app.command("render")(render_cmd.render)
    app.command("review")(render_cmd.review)
    app.command("pick")(render_cmd.pick)
    app.command("conform")(render_cmd.conform)
    app.add_typer(audio_cmd.audio_app, name="audio")
    app.add_typer(audio_cmd.lyrics_app, name="lyrics")
    app.add_typer(audio_cmd.voice_app, name="voice")
    app.command("doctor")(doctor_cmd.doctor)
    app.add_typer(config_cmd.app, name="config")
    app.add_typer(engines_cmd.app, name="engines")
    app.add_typer(workflows_cmd.app, name="workflows")
    app.add_typer(models_cmd.app, name="models")
    app.add_typer(worker_cmd.app, name="worker")
    app.command("bench")(bench_cmd.bench)
