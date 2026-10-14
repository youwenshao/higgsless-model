"""`higgsless char ...` and `higgsless keyframe ...`."""

from __future__ import annotations

from pathlib import Path

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.pipeline import images
from higgsless.pipeline.stage import Stage

char_app = typer.Typer(help="Characters: identity portrait, reference sheet, extra angles.", no_args_is_help=True)
keyframe_app = typer.Typer(help="Per-shot sample-scene keyframes and their review gate.", no_args_is_help=True)


@char_app.command("create")
def char_create(
    ctx: typer.Context,
    name: str,
    prompt: str | None = typer.Option(None, help="Appearance description (used alone or with images)."),
    image: list[Path] = typer.Option([], "--image", "-i", help="Reference image(s) of the character (up to 4)."),
    style: str | None = typer.Option(None, help="Rendering style, e.g. 'anime cel shading'."),
    stylised: bool | None = typer.Option(None, "--stylised/--photoreal", help="Routes lip-sync to InfiniteTalk."),
    angles: bool = typer.Option(False, help="Also render left/right/back views with the multi-angle LoRA."),
    seed: int | None = typer.Option(None),
    force: bool = typer.Option(False, help="Regenerate even if cached."),
) -> None:
    """Create or update a character and render its reference sheet."""
    app = get_ctx(ctx)
    result = images.create_character(
        Stage(app),
        name,
        prompt=prompt,
        images=image,
        style=style,
        stylised=stylised,
        angles=angles,
        seed=seed,
        force=force,
    )
    app.out.result(result, f"{name}: sheet ready; review {result['contact_sheet']}")


@char_app.command("list")
def char_list(ctx: typer.Context) -> None:
    """List characters with their sheets."""
    app = get_ctx(ctx)
    project = app.project.load()
    data = {n: c.model_dump(mode="json") for n, c in project.characters.items()}
    rows = [
        [n, (c["prompt"] or "")[:40], c["sheet"] or "(none)", "yes" if c["stylised"] else ""] for n, c in data.items()
    ]
    app.out.result(data, table("characters", ["name", "prompt", "sheet", "stylised"], rows))


@keyframe_app.command("make")
def keyframe_make(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(None, help="Shots (default: all that will be rendered)."),
    seed: int | None = typer.Option(None),
    force: bool = typer.Option(False, help="Regenerate (new seed if --seed is not given)."),
) -> None:
    """Edit each shot's first frame into a sample scene with the cast characters, then caption it."""
    app = get_ctx(ctx)
    stage = Stage(app)
    targets = shot_ids or [s.id for s in stage.project.shots]
    results = {sid: images.make_keyframe(stage, sid, seed=seed, force=force) for sid in targets}
    sheet = images.keyframes_contact_sheet(stage)
    rows = [
        [sid, r.get("mode", r.get("skipped")), r.get("engine") or "", "yes" if r.get("approved") else ""]
        for sid, r in results.items()
    ]
    app.out.result(
        {"keyframes": results, "contact_sheet": sheet}, table("keyframes", ["shot", "mode", "engine", "approved"], rows)
    )


@keyframe_app.command("approve")
def keyframe_approve(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Approve every keyframe."),
) -> None:
    """Approve keyframes so their shots can render."""
    app = get_ctx(ctx)
    stage = Stage(app)
    targets = [s.id for s in stage.project.shots] if all_ else (shot_ids or [])
    changed = images.set_approval(stage, targets, True)
    images.keyframes_contact_sheet(stage)
    app.out.result({"approved": changed}, f"approved {len(changed)} keyframe(s)")


@keyframe_app.command("reject")
def keyframe_reject(ctx: typer.Context, shot_ids: list[str], note: str | None = typer.Option(None)) -> None:
    """Reject keyframes (then rerun `keyframe make <id> --force`, optionally after `shots set --prompt`)."""
    app = get_ctx(ctx)
    changed = images.set_approval(Stage(app), shot_ids, False, note)
    app.out.result({"rejected": changed}, f"rejected {len(changed)} keyframe(s)")
