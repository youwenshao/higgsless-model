"""`higgsless audio ...`, `lyrics` and `voice`."""

from __future__ import annotations

from pathlib import Path

import typer

from higgsless.cli.context import get_ctx
from higgsless.core.manifest import Consent
from higgsless.pipeline import audio as audio_stage
from higgsless.pipeline.stage import Stage

audio_app = typer.Typer(help="Separate, transcribe, sing and mix.", no_args_is_help=True)
voice_app = typer.Typer(help="The voice that sings the new lyrics.", no_args_is_help=True)
lyrics_app = typer.Typer(help="Check new lyrics against the transcribed melody.", no_args_is_help=True)


@audio_app.command("separate")
def separate(ctx: typer.Context, force: bool = typer.Option(False)) -> None:
    """Split the song into instrumental, lead, backing and a dry lead."""
    app = get_ctx(ctx)
    state = audio_stage.separate(Stage(app), force=force)
    app.out.result(state["stems"], f"{len(state['stems'])} stems")


@audio_app.command("transcribe")
def transcribe(ctx: typer.Context, force: bool = typer.Option(False)) -> None:
    """Transcribe the dry lead into notes and word-timed lyrics, and write a starter lyrics file."""
    app = get_ctx(ctx)
    result = audio_stage.transcribe(Stage(app), force=force)
    app.out.result(result, f"{result['segments']} segments; edit {result['lyrics_file']}")


@lyrics_app.command("fit")
def lyrics_fit(ctx: typer.Context) -> None:
    """Check that each new lyric line has one word (or character) per sung note."""
    app = get_ctx(ctx)
    report = audio_stage.fit_lyrics(Stage(app))
    app.out.result(
        report,
        "lyrics fit the melody" if report["ok"] else f"{len(report['mismatches'])} line(s) do not match the notes",
    )


@audio_app.command("sing")
def sing(ctx: typer.Context, force: bool = typer.Option(False)) -> None:
    """Sing audio/lyrics_new.yaml in the lead voice, locked to the original melody."""
    app = get_ctx(ctx)
    state = audio_stage.sing(Stage(app), force=force)
    app.out.result({"lead_new": state["lead_new"]}, f"sang {state['lead_new']}")


@audio_app.command("mix")
def mix(ctx: typer.Context, force: bool = typer.Option(False)) -> None:
    """Put the new lead back on the instrumental and master it against the original track."""
    app = get_ctx(ctx)
    state = audio_stage.mix(Stage(app), force=force)
    app.out.result({"final_mix": state["final_mix"]}, f"mixed {state['final_mix']}")


@voice_app.command("create")
def voice_create(
    ctx: typer.Context,
    name: str,
    kind: str = typer.Option("designed", help="designed (invented) or reference (a clip of a consenting singer)."),
    describe: str | None = typer.Option(None, help="For a designed voice: 'warm alto pop vocalist'."),
    reference: Path | None = typer.Option(None, help="10-30 s of clean solo singing."),
    consent: Consent = typer.Option(Consent.SYNTHETIC, help="self, written or synthetic."),
    text: str | None = typer.Option(None, help="Sentence the designed voice speaks."),
    force: bool = typer.Option(False),
) -> None:
    """Create the lead voice and select it."""
    app = get_ctx(ctx)
    stage = Stage(app)
    if kind == "designed":
        if not describe:
            raise typer.BadParameter("a designed voice needs --describe")
        result = audio_stage.design_voice(stage, name, describe, text=text, force=force)
    elif kind == "reference":
        if reference is None:
            raise typer.BadParameter("a reference voice needs --reference")
        result = audio_stage.set_reference_voice(stage, name, reference, consent)
    else:
        raise typer.BadParameter("kind is designed or reference (train an RVC model separately)")
    app.out.result(result, f"voice {name} is the lead")
