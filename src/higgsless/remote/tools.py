"""Run worker tasks and tool containers on the workstation.

Worker tasks are modules in `higgsless.worker_tasks`. They run inside the `tools` container (or
natively when tools_mode is "native"), take `--result <path>` and write a JSON result there.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from higgsless.core.config import WorkstationConfig
from higgsless.core.project import remote_join
from higgsless.errors import JobFailed
from higgsless.remote.comfy import ComfyClient
from higgsless.remote.executor import CommandResult, Executor, LineCallback

CONTAINER_WORKER = "/opt/higgsless"


class ToolRunner:
    def __init__(self, cfg: WorkstationConfig, executor: Executor, comfy: ComfyClient | None = None) -> None:
        self.cfg = cfg
        self.executor = executor
        self.comfy = comfy
        self._user: str | None = cfg.docker.user

    def _docker_user(self) -> str | None:
        if self._user is None and self.executor.kind == "ssh":
            uid = self.executor.run(["id", "-u"]).stdout.strip()
            gid = self.executor.run(["id", "-g"]).stdout.strip()
            self._user = f"{uid}:{gid}"
        return self._user

    def native(self, image_key: str) -> bool:
        return self.cfg.tools_mode == "native" and image_key == "tools"

    def argv(self, image_key: str, cmd: list[str], *, gpu: bool, env: dict[str, str] | None = None) -> list[str]:
        if self.native(image_key):
            return cmd
        cfg = self.cfg
        try:
            image = cfg.docker.images[image_key]
        except KeyError as exc:
            raise JobFailed(f"no docker image configured for {image_key!r}") from exc
        argv = ["docker", "run", "--rm", "--init", "--ipc=host"]
        if gpu:
            argv += ["--gpus", cfg.docker.gpus]
        user = self._docker_user()
        if user:
            argv += ["--user", user, "-e", "HOME=/tmp"]
        argv += [
            "-v",
            f"{cfg.data_root}:{cfg.data_root}",
            "-v",
            f"{cfg.hf_cache}:/hf",
            "-v",
            f"{cfg.worker_dir}:{CONTAINER_WORKER}:ro",
            "-e",
            "HF_HOME=/hf",
            "-e",
            f"PYTHONPATH={CONTAINER_WORKER}/src",
            "-e",
            f"HIGGSLESS_DATA_ROOT={cfg.data_root}",
        ]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv += [*cfg.docker.extra_args, image, *cmd]
        return argv

    def _python(self, image_key: str) -> str:
        return self.cfg.native_python if self.native(image_key) else "python"

    def _native_env(self, image_key: str, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self.native(image_key):
            return None
        merged = {"PYTHONPATH": remote_join(self.cfg.worker_dir, "src"), "HIGGSLESS_DATA_ROOT": self.cfg.data_root}
        merged.update(env or {})
        return merged

    def _prepare_gpu(self, gpu: bool) -> None:
        if gpu and self.comfy is not None:
            # Containers share the GPU with ComfyUI; unload its models first.
            self.comfy.free()

    def run_task(
        self,
        task: str,
        args: list[str],
        *,
        gpu: bool = False,
        image: str = "tools",
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_line: LineCallback | None = None,
    ) -> dict[str, Any]:
        result_path = remote_join(self.cfg.data_root, "_tmp", f"{task}-{uuid.uuid4().hex[:12]}.json")
        self.executor.makedirs(remote_join(self.cfg.data_root, "_tmp"))
        cmd = [self._python(image), "-m", f"higgsless.worker_tasks.{task}", "--result", result_path, *args]
        self._prepare_gpu(gpu)
        self.executor.run(
            self.argv(image, cmd, gpu=gpu, env=env),
            env=self._native_env(image, env),
            timeout=timeout,
            on_line=on_line,
        )
        try:
            data = json.loads(self.executor.read_text(result_path))
        except FileNotFoundError as exc:
            raise JobFailed(f"worker task {task} wrote no result") from exc
        finally:
            self.executor.remove(result_path)
        if not data.get("ok", True):
            raise JobFailed(f"worker task {task} failed: {data.get('error')}", details=data)
        return data

    def run_container(
        self,
        image: str,
        cmd: list[str],
        *,
        gpu: bool = True,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_line: LineCallback | None = None,
    ) -> CommandResult:
        self._prepare_gpu(gpu)
        return self.executor.run(
            self.argv(image, cmd, gpu=gpu, env=env),
            env=self._native_env(image, env),
            timeout=timeout,
            on_line=on_line,
        )
