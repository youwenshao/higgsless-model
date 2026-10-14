"""`higgsless shots ...`: detect, inspect and edit shots."""

from __future__ import annotations

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.core.manifest import Mode, Switch
from higgsless.core.timecode import frame_to_timecode, parse_rate
from higgsless.errors import ValidationError
from higgsless.pipeline import shots as shots_stage
from higgsless.pipeline.stage import Stage

app = typer.Typer(help="Detect, inspect and edit shots.", no_args_is_help=True)


@app.command("detect")
def detect(
    ctx: typer.Context,
    min_frames: int = typer.Option(8, help="Shortest allowed shot."),
    adaptive: float = typer.Option(3.0, help="PySceneDetect adaptive threshold."),
    content: float = typer.Option(27.0, help="PySceneDetect content threshold."),
    transnet: bool = typer.Option(True, help="Also use TransNetV2 when installed."),
    force: bool = typer.Option(False, help="Re-detect (settings of unchanged shots are kept)."),
) -> None:
    """Detect shot boundaries, cut per-shot mezzanines and write a contact sheet."""
    app_ctx = get_ctx(ctx)
    summary = shots_stage.detect(Stage(app_ctx), min_frames=min_frames, adaptive=adaptive, content=content,
                                 transnet=transnet, force=force)
    app_ctx.out.result(summary, f"{summary['count']} shots; contact sheet: {summary['contact_sheet']}")


@app.command("list")
def list_shots(ctx: typer.Context) -> None:
    """List shots with timing, mode, engine and analysis summary."""
    app_ctx = get_ctx(ctx)
    handle = app_ctx.project
    project = handle.load()
    rate = parse_rate(project.source.rate) if project.source else None
    data = []
    for s in project.shots:
        analysis = handle.state.read(f"analysis/{s.id}") or {}
        data.append({
            "id": s.id, "frames": s.frames, "length": s.length,
            "start_tc": frame_to_timecode(s.start, rate) if rate else None,
            "mode": s.mode.value, "engine": s.engine, "lipsync": s.lipsync.value, "transition": s.transition,
            "people": len(analysis.get("people", [])) if analysis else None,
            "face_frac": (analysis.get("faces") or {}).get("max_frac") if analysis else None,
        })
    rows = [[d["id"], d["start_tc"], d["length"], d["mode"], d["engine"] or "", d["people"], d["lipsync"]] for d in data]
    app_ctx.out.result(data, table("shots", ["id", "start", "frames", "mode", "engine", "people", "lipsync"], rows))


@app.command("set")
def set_shot(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(..., help="Shot ids (or 'all')."),
    mode: Mode | None = typer.Option(None),
    engine: str | None = typer.Option(None, help="Engine id, or 'none' to clear."),
    lipsync: Switch | None = typer.Option(None),
    prompt: str | None = typer.Option(None, help="Extra prompt for this shot ('none' clears)."),
    takes: int | None = typer.Option(None),
    cast: list[str] = typer.Option([], help="track=character, e.g. 1=ava (repeatable)."),
    notes: str | None = typer.Option(None),
) -> None:
    """Set per-shot options in project.yaml."""
    app_ctx = get_ctx(ctx)
    handle = app_ctx.project
    known = [s.id for s in handle.load().shots]
    targets = known if shot_ids == ["all"] else shot_ids
    unknown = sorted(set(targets) - set(known))
    if unknown:
        raise ValidationError(f"unknown shot ids: {unknown}")
    with handle.edit() as data:
        for entry in data["shots"]:
            if entry["id"] not in targets:
                continue
            if mode is not None:
                entry["mode"] = mode.value
            if engine is not None:
                entry.pop("engine", None) if engine == "none" else entry.__setitem__("engine", engine)
            if lipsync is not None:
                entry["lipsync"] = lipsync.value
            if prompt is not None:
                entry.pop("prompt", None) if prompt == "none" else entry.__setitem__("prompt", prompt)
            if takes is not None:
                entry["takes"] = takes
            if notes is not None:
                entry["notes"] = notes
            for item in cast:
                track, _, character = item.partition("=")
                entry.setdefault("cast", {})[track] = character
    app_ctx.out.result({"updated": targets}, f"updated {len(targets)} shot(s)")


@app.command("split")
def split(ctx: typer.Context, shot_id: str, at: int = typer.Option(..., help="Absolute source frame to split at.")) -> None:
    """Split a shot in two at a source frame (the new shot gets a letter suffix)."""
    app_ctx = get_ctx(ctx)
    ids = shots_stage.split(Stage(app_ctx), shot_id, at)
    app_ctx.out.result({"shots": ids}, f"split into {ids}")


@app.command("merge")
def merge(ctx: typer.Context, first: str, second: str) -> None:
    """Merge two adjacent shots."""
    app_ctx = get_ctx(ctx)
    kept = shots_stage.merge(Stage(app_ctx), first, second)
    app_ctx.out.result({"shot": kept}, f"merged into {kept}")
