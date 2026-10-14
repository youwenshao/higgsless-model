"""Root of the `higgsless` CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import typer

from higgsless import __version__
from higgsless.cli.context import AppContext, get_ctx
from higgsless.cli.output import Output
from higgsless.errors import HiggslessError

app = typer.Typer(
    name="higgsless",
    help=(
        "Regenerate a music video with new characters and new vocals on local open models.\n\n"
        "Every command is idempotent, non-interactive and supports --json for agents."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

_ROOT_CTX: AppContext | None = None


@app.callback()
def _root(
    ctx: typer.Context,
    project: str | None = typer.Option(
        None, "--project", "-p", envvar="HIGGSLESS_PROJECT", help="Project name or path to project.yaml."
    ),
    json_out: bool = typer.Option(False, "--json", envvar="HIGGSLESS_JSON", help="Emit one JSON document."),
    config: Path | None = typer.Option(
        None, "--config", envvar="HIGGSLESS_CONFIG", help="Workstation config (default config/workstation.yaml)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="More progress output on stderr."),
) -> None:
    global _ROOT_CTX
    _ROOT_CTX = AppContext(
        project_selector=project,
        config_path=config,
        verbose=verbose,
        out=Output(json_mode=json_out),
    )
    ctx.obj = _ROOT_CTX
    ctx.call_on_close(_ROOT_CTX.close)


@app.command()
def version(ctx: typer.Context) -> None:
    """Print the higgsless version."""
    get_ctx(ctx).out.result({"version": __version__}, f"higgsless {__version__}")


def _register() -> None:
    from higgsless.cli import commands

    commands.register(app)


def _fallback_output() -> Output:
    if _ROOT_CTX is not None:
        return _ROOT_CTX.out
    json_mode = "--json" in sys.argv or os.environ.get("HIGGSLESS_JSON", "").lower() in {"1", "true", "yes"}
    return Output(json_mode=json_mode)


def main() -> None:
    _register()
    try:
        rv = app(standalone_mode=False)
    except HiggslessError as err:
        _fallback_output().error(err)
        sys.exit(err.exit_code)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.exceptions.ClickException as err:
        err.show()
        sys.exit(err.exit_code)
    except click.exceptions.Exit as err:
        sys.exit(err.exit_code)
    sys.exit(rv if isinstance(rv, int) else 0)
