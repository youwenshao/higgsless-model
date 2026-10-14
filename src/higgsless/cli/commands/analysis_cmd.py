"""`higgsless analyze` and `higgsless cast ...`."""

from __future__ import annotations

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.errors import ValidationError
from higgsless.pipeline import analyze as analysis
from higgsless.pipeline.stage import Stage

cast_app = typer.Typer(help="Performers found across shots, and who replaces them.", no_args_is_help=True)


def analyze(
    ctx: typer.Context,
    shot_ids: list[str] = typer.Argument(None, help="Shots to analyse (default: all)."),
    captions: bool = typer.Option(True, help="Caption each shot with Qwen3-VL."),
    force: bool = typer.Option(False, help="Redo finished analysis."),
) -> None:
    """Track people (SAM 3.1), measure faces and camera motion, and caption shots."""
    app = get_ctx(ctx)
    result = analysis.analyze(Stage(app), shot_ids or None, captions=captions, force=force)
    rows = [[sid, d["people"], d["faces"], d["motion"]] for sid, d in result.items()]
    app.out.result(result, table("analysis", ["shot", "people", "max face", "motion"], rows))


@cast_app.command("detect")
def cast_detect(ctx: typer.Context, threshold: float = typer.Option(0.72, help="Cosine similarity to merge.")) -> None:
    """Cluster tracked people across shots into performers (P1 = most frequent)."""
    app = get_ctx(ctx)
    state = analysis.detect_cast(Stage(app), threshold=threshold)
    rows = [[pid, len(p["members"]), ", ".join(p["shots"][:8])] for pid, p in state["performers"].items()]
    app.out.result(state, table("performers", ["id", "appearances", "shots"], rows))


@cast_app.command("list")
def cast_list(ctx: typer.Context) -> None:
    """Show performers and the character each is cast as."""
    app = get_ctx(ctx)
    project = app.project.load()
    state = app.project.state.read("cast") or {"performers": {}}
    data = {
        pid: {"character": project.cast.get(pid), "shots": p["shots"], "appearances": len(p["members"])}
        for pid, p in state["performers"].items()
    }
    rows = [[pid, d["character"] or "(unassigned)", d["appearances"]] for pid, d in data.items()]
    app.out.result(
        {"performers": data, "contact_sheet": state.get("contact_sheet")},
        table("cast", ["performer", "character", "appearances"], rows),
    )


@cast_app.command("assign")
def cast_assign(ctx: typer.Context, performer: str, character: str) -> None:
    """Cast a performer (P1...) as a character defined in project.yaml ('none' unassigns)."""
    app = get_ctx(ctx)
    state = app.project.state.read("cast") or {"performers": {}}
    if performer not in state["performers"]:
        raise ValidationError(f"unknown performer {performer}", hint=f"known: {sorted(state['performers'])}")
    with app.project.edit() as data:
        if character == "none":
            data["cast"].pop(performer, None)
        else:
            data["cast"][performer] = character
    app.out.result({"performer": performer, "character": character}, f"{performer} -> {character}")
