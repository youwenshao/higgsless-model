"""`higgsless init | ingest | validate`."""

from __future__ import annotations

from pathlib import Path

import typer

from higgsless.cli.context import get_ctx
from higgsless.core.manifest import Look, Profile, Usage
from higgsless.core.project import create_project
from higgsless.pipeline.ingest import ingest as run_ingest
from higgsless.pipeline.stage import Stage


def init(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name (lowercase, digits, - and _)."),
    source: Path | None = typer.Option(None, "--source", "-s", help="Source music video to ingest."),
    usage: Usage = typer.Option(Usage.PUBLIC_NONCOMMERCIAL, help="Decides which engine licences are allowed."),
    look: Look = typer.Option(Look.NEW_WORLD, help="original | new-world | restyle"),
    profile: Profile = typer.Option(Profile.STANDARD, help="draft | standard | max"),
    exist_ok: bool = typer.Option(False, "--exist-ok", help="Reuse an existing project of this name."),
) -> None:
    """Create a project (and ingest the source video if given)."""
    app = get_ctx(ctx)
    handle = create_project(name, usage=usage.value, look=look.value, profile=profile.value, exist_ok=exist_ok)
    app._project = handle
    data: dict = {"project": name, "root": str(handle.root), "manifest": str(handle.manifest_path)}
    if source is not None:
        data["ingest"] = run_ingest(Stage(app), source)
    app.out.result(data, f"created {handle.manifest_path}")


def ingest(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="Source music video."),
    force: bool = typer.Option(False, help="Re-upload even if the remote copy matches."),
) -> None:
    """Upload and probe the source video, then make the proxy and extract the original audio."""
    app = get_ctx(ctx)
    state = run_ingest(Stage(app), source, force=force)
    probe = state["probe"]
    app.out.result(state, f"{probe['frames']} frames at {probe['rate']} fps, {probe['width']}x{probe['height']}")


def validate(ctx: typer.Context) -> None:
    """Validate project.yaml (schema, cast, shots, voices)."""
    app = get_ctx(ctx)
    project = app.project.load(refresh=True)
    app.out.result(
        {
            "project": project.project,
            "shots": len(project.shots),
            "characters": sorted(project.characters),
            "voices": sorted(project.voices),
            "look": project.look.value,
            "profile": project.profile.value,
        },
        f"{app.project.manifest_path} is valid",
    )
