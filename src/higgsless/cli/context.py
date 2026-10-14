"""Per-invocation context shared by all commands (global flags, lazily created clients)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from higgsless.cli.output import Output

if TYPE_CHECKING:
    from higgsless.core.config import WorkstationConfig
    from higgsless.core.project import ProjectHandle
    from higgsless.remote.comfy import ComfyClient
    from higgsless.remote.executor import Executor


@dataclass
class AppContext:
    project_selector: str | None = None
    config_path: Path | None = None
    verbose: bool = False
    out: Output = field(default_factory=Output)
    _config: Any = None
    _project: Any = None
    _executor: Any = None
    _comfy: Any = None

    @property
    def config(self) -> WorkstationConfig:
        if self._config is None:
            from higgsless.core.config import load_config

            self._config = load_config(self.config_path)
        return self._config

    @property
    def project(self) -> ProjectHandle:
        if self._project is None:
            from higgsless.core.project import open_project

            self._project = open_project(self.project_selector)
        return self._project

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            from higgsless.remote.executor import make_executor

            self._executor = make_executor(self.config)
        return self._executor

    @property
    def comfy(self) -> ComfyClient:
        if self._comfy is None:
            from higgsless.remote.comfy import ComfyClient

            self._comfy = ComfyClient(self.config.comfy.url, timeout_s=self.config.comfy.timeout_s)
        return self._comfy

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()
        if self._comfy is not None:
            self._comfy.close()


def get_ctx(ctx: typer.Context) -> AppContext:
    obj = ctx.find_root().obj
    if not isinstance(obj, AppContext):  # pragma: no cover - set by the root callback
        raise RuntimeError("root callback did not run")
    path = ctx.command_path
    obj.out.command = path.split("higgsless", 1)[1].strip() if "higgsless" in path else path
    return obj
