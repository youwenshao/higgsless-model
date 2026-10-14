"""Helpers shared by pipeline stages: running worker tasks for a project, pulling previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from higgsless.core.config import WorkstationConfig
from higgsless.core.manifest import Project
from higgsless.core.project import ProjectHandle, remote_join
from higgsless.core.timecode import FramePolicy, parse_rate, plan_frame_policy
from higgsless.errors import ConfigError
from higgsless.remote.deploy import deploy_worker
from higgsless.remote.executor import Executor
from higgsless.remote.tools import ToolRunner


@dataclass
class Stage:
    """Everything a stage needs: config, executor, project handle, tool runner, output."""

    app: Any  # AppContext

    @property
    def cfg(self) -> WorkstationConfig:
        return self.app.config

    @property
    def ex(self) -> Executor:
        return self.app.executor

    @property
    def handle(self) -> ProjectHandle:
        return self.app.project

    @property
    def project(self) -> Project:
        return self.handle.load()

    def rpath(self, *parts: str) -> str:
        return self.handle.rpath(self.cfg, *parts)

    def input_rel(self, path: str) -> str:
        """Remote absolute path -> path relative to ComfyUI's input directory (the data root)."""
        root = self.cfg.data_root.replace("\\", "/").rstrip("/")
        norm = str(path).replace("\\", "/")
        windows = len(root) > 1 and root[1] == ":"
        prefix_ok = (norm.lower() if windows else norm).startswith((root.lower() if windows else root) + "/")
        if not prefix_ok:
            raise ConfigError(f"{path} is outside the data root {root}")
        return norm[len(root) + 1 :]

    def progress(self, message: str) -> None:
        self.app.out.progress(message)

    def ensure_worker(self) -> None:
        if not getattr(self.app, "_worker_checked", False):
            if self.cfg.tools_mode == "docker" or self.cfg.executor == "ssh":
                deploy_worker(self.ex, self.cfg.worker_dir)
            self.app._worker_checked = True

    def task(
        self,
        name: str,
        params: dict[str, Any],
        output_dir: str,
        *,
        gpu: bool = False,
        image: str = "tools",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.ensure_worker()
        runner = ToolRunner(self.cfg, self.ex, self.app.comfy if gpu else None)
        return runner.run_task(
            name,
            ["--output-dir", output_dir],
            params=params,
            gpu=gpu,
            image=image,
            timeout=timeout,
            on_line=lambda line: self.progress(line) if line.strip() and self.app.verbose else None,
        )

    def pull(self, remote: str, name: str) -> Path:
        """Copy a small file (preview, contact sheet) into projects/<name>/previews/."""
        local = self.handle.previews_dir / name
        self.ex.get(remote, local)
        return local

    def require_source(self) -> Any:
        source = self.project.source
        if source is None:
            raise ConfigError("the project has no source yet", hint="higgsless ingest <video>")
        return source

    def frame_policy(self) -> FramePolicy:
        return plan_frame_policy(parse_rate(self.require_source().rate))

    def shot_dir(self, shot_id: str) -> str:
        return self.rpath("shots", shot_id)

    def join(self, root: str, *parts: str) -> str:
        return remote_join(root, *parts)
