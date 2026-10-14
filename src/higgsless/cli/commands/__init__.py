"""Command registration. Each module exposes `register(app)` or a typer sub-app."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    pass
