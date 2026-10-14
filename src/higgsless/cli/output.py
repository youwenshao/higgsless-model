"""Output helpers: one JSON document on stdout in --json mode, rich text otherwise.

Progress always goes to stderr so stdout stays parseable for agents.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from higgsless.core.jsonio import to_json
from higgsless.errors import HiggslessError


@dataclass
class Output:
    json_mode: bool = False
    command: str = ""
    stdout: Console = field(default_factory=lambda: Console(soft_wrap=True))
    stderr: Console = field(default_factory=lambda: Console(stderr=True, soft_wrap=True))

    def progress(self, message: str) -> None:
        self.stderr.print(f"[dim]{message}[/dim]", highlight=False)

    def warn(self, message: str) -> None:
        self.stderr.print(f"[yellow]warning:[/yellow] {message}", highlight=False)

    def result(self, data: Any, human: str | Table | None = None) -> None:
        """Emit the command's final result."""
        if self.json_mode:
            sys.stdout.write(to_json({"ok": True, "command": self.command, "data": data}) + "\n")
            sys.stdout.flush()
            return
        if human is None:
            self.stdout.print_json(to_json(data))
        else:
            self.stdout.print(human, highlight=False)

    def error(self, err: HiggslessError) -> None:
        if self.json_mode:
            payload = {"ok": False, "command": self.command, "error": err.to_dict()}
            sys.stdout.write(to_json(payload) + "\n")
            sys.stdout.flush()
            return
        self.stderr.print(f"[red]error ({err.kind}):[/red] {err.message}", highlight=False)
        if err.hint:
            self.stderr.print(f"[cyan]hint:[/cyan] {err.hint}", highlight=False)
        if err.details is not None:
            self.stderr.print_json(to_json(err.details))


def table(title: str, columns: list[str], rows: list[list[Any]]) -> Table:
    t = Table(title=title, show_lines=False)
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*["" if v is None else str(v) for v in r])
    return t
