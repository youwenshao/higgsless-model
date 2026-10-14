"""`higgsless render | review | pick | conform`."""

from __future__ import annotations

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.pipeline import render as render_stage
from higgsless.pipeline.conform import conform as run_conform
from higgsless.pipeline.stage import Stage


def _targets(stage: Stage, shot_ids: list[str] | None) -> list[str]:
    return shot_ids or [s.id for s in stage.project.shots]


def render(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(None, help="Shots to render (default: all)."),
    takes: int | None = typer.Option(None, help="Takes per shot (default: shot or project setting)."),
    force: bool = typer.Option(False, help="Re-render finished windows."),
    gate: bool = typer.Option(True, help="Require approved keyframes."),
) -> None:
    """Render shots window by window with the routed engine; resumable."""
    app = get_ctx(ctx)
    stage = Stage(app)
    results = {}
    for shot_id in _targets(stage, shot_ids):
        app.out.progress(f"rendering {shot_id}")
        results[shot_id] = render_stage.render_shot(stage, shot_id, takes=takes, force=force, gate=gate)
    rows = [
        [sid, r.get("mode"), r.get("engine") or "", len(r.get("takes") or {}), r.get("picked") or ""]
        for sid, r in results.items()
    ]
    app.out.result(results, table("renders", ["shot", "mode", "engine", "takes", "picked"], rows))


def review(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(None, help="Shots to review (default: all rendered)."),
    force: bool = typer.Option(False, help="Recompute metrics and sheets."),
) -> None:
    """Measure takes (frame count, flicker vs source, identity vs keyframe) and write review sheets."""
    app = get_ctx(ctx)
    stage = Stage(app)
    data = {}
    rows = []
    for shot_id in _targets(stage, shot_ids):
        state = render_stage.review_shot(stage, shot_id, force=force)
        data[shot_id] = state
        for take, info in (state.get("takes") or {}).items():
            m = info.get("metrics") or {}
            ident = (m.get("identity") or {}).get("mean")
            rows.append(
                [
                    shot_id,
                    take,
                    "*" if state.get("picked") == take else "",
                    m.get("frames"),
                    m.get("flicker_ratio"),
                    ident,
                    info.get("review_sheet", ""),
                ]
            )
    app.out.result(data, table("review", ["shot", "take", "picked", "frames", "flicker x", "identity", "sheet"], rows))


def pick(ctx: typer.Context, shot_id: str, take: int = typer.Option(..., help="Take number.")) -> None:
    """Choose which take of a shot goes into the conform."""
    app = get_ctx(ctx)
    state = render_stage.pick(Stage(app), shot_id, take)
    app.out.result({"shot": shot_id, "picked": state["picked"]}, f"{shot_id}: take {take}")


def conform(
    ctx: typer.Context,
    audio: str = typer.Option("auto", help="auto (final mix if present) | mix | original | none"),
    resolution: str | None = typer.Option(None, help="source | 720p | 1080p | 2160p (default: project delivery)"),
    pull: bool = typer.Option(False, help="Copy the final video and OTIO timeline to previews/."),
    name: str = typer.Option("final.mp4"),
) -> None:
    """Reassemble all shots frame-exactly at the source rate and mux the audio; also writes an OTIO timeline."""
    app = get_ctx(ctx)
    result = run_conform(Stage(app), audio=audio, resolution=resolution, pull=pull, name=name)
    app.out.result(result, f"{result['video']} ({result['frames']} frames, {result['duration_s']} s)")
