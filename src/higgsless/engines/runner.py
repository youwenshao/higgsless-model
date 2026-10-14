"""Run engine jobs (ComfyUI workflows or worker tasks) with caching and job records."""

from __future__ import annotations

import json
import posixpath
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import Any

from higgsless.core.cache import hash_obj, sha256_text
from higgsless.core.config import WorkstationConfig
from higgsless.core.paths import data_dir
from higgsless.core.project import ProjectHandle, remote_join
from higgsless.core.state import StateStore
from higgsless.engines.bindings import Bindings, bind
from higgsless.engines.spec import EngineSpec
from higgsless.errors import HiggslessError, JobFailed
from higgsless.remote.comfy import ComfyClient
from higgsless.remote.executor import Executor
from higgsless.remote.tools import ToolRunner


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@cache
def _workflow_text(rel: str) -> str:
    return (data_dir("workflows") / rel).read_text(encoding="utf-8")


def load_workflow(engine: EngineSpec) -> tuple[dict[str, Any], Bindings]:
    assert engine.workflow and engine.bindings
    workflow = json.loads(_workflow_text(engine.workflow))
    bindings = Bindings.load(data_dir("workflows") / engine.bindings)
    return workflow, bindings


def engine_fingerprint(engine: EngineSpec) -> str:
    parts: dict[str, Any] = {"spec": engine.model_dump(mode="json")}
    if engine.backend == "comfyui":
        parts["workflow"] = sha256_text(_workflow_text(engine.workflow or ""))
        parts["bindings"] = sha256_text(_workflow_text(engine.bindings or ""))
    return hash_obj(parts)


@dataclass
class JobSpec:
    engine: EngineSpec
    params: dict[str, Any]
    output_dir: str  # absolute remote directory for this job's outputs
    label: str
    inputs: dict[str, str] = field(default_factory=dict)  # name -> fingerprint (upstream job key or sha256)
    timeout_s: float | None = None

    def key(self) -> str:
        return hash_obj(
            {
                "engine": self.engine.id,
                "engine_fp": engine_fingerprint(self.engine),
                "params": self.params,
                "inputs": self.inputs,
                "output_dir": self.output_dir,
            }
        )


class JobRunner:
    def __init__(
        self,
        cfg: WorkstationConfig,
        executor: Executor,
        comfy: ComfyClient,
        state: StateStore,
        progress: Any = None,
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.comfy = comfy
        self.state = state
        self.tools = ToolRunner(cfg, executor, comfy)
        self.progress = progress or (lambda msg: None)

    def _output_rel(self, output_dir: str) -> str:
        root = self.cfg.data_root.rstrip("/\\")
        rel = output_dir[len(root) :].lstrip("/\\") if output_dir.startswith(root) else output_dir
        return rel.replace("\\", "/")

    def cached(self, spec: JobSpec) -> dict[str, Any] | None:
        record = self.state.read(f"jobs/{spec.key()}")
        if not record or record.get("status") != "done":
            return None
        for paths in (record.get("outputs") or {}).values():
            for path in paths[:1] + paths[-1:]:
                if not self.executor.exists(path):
                    return None
        return record

    def run(self, spec: JobSpec, *, force: bool = False) -> dict[str, Any]:
        key = spec.key()
        if not force and (record := self.cached(spec)) is not None:
            return {**record, "cached": True}
        record: dict[str, Any] = {
            "key": key,
            "engine": spec.engine.id,
            "label": spec.label,
            "params": spec.params,
            "inputs": spec.inputs,
            "output_dir": spec.output_dir,
            "status": "running",
            "started_at": _now(),
        }
        self.state.write(f"jobs/{key}", record)
        start = time.monotonic()
        try:
            self.executor.makedirs(spec.output_dir)
            if spec.engine.backend == "comfyui":
                outputs, extra = self._run_comfy(spec)
            else:
                outputs, extra = self._run_task(spec)
        except HiggslessError as exc:
            record.update(
                status="failed", error=exc.to_dict(), finished_at=_now(), duration_s=round(time.monotonic() - start, 2)
            )
            self.state.write(f"jobs/{key}", record)
            raise
        record.update(
            status="done", outputs=outputs, finished_at=_now(), duration_s=round(time.monotonic() - start, 2), **extra
        )
        self.state.write(f"jobs/{key}", record)
        return {**record, "cached": False}

    def _run_comfy(self, spec: JobSpec) -> tuple[dict[str, list[str]], dict[str, Any]]:
        workflow, bindings = load_workflow(spec.engine)
        rel = self._output_rel(spec.output_dir)
        prefixes = {name: posixpath.join(rel, name) for name in bindings.outputs}
        bound = bind(workflow, bindings, spec.params, prefixes)
        last = {"t": 0.0}

        def on_progress(evt: dict[str, Any]) -> None:
            now = time.monotonic()
            if evt.get("max") and now - last["t"] > 5:
                last["t"] = now
                self.progress(f"{spec.label}: node {evt['node']} step {evt['value']}/{evt['max']}")

        self.progress(f"{spec.label}: queued on ComfyUI ({spec.engine.id})")
        prompt_id, entry = self.comfy.run(bound.workflow, on_progress, timeout_s=spec.timeout_s)
        by_node = {node_id: name for name, node_id in bound.outputs.items()}
        outputs: dict[str, list[str]] = {name: [] for name in bound.outputs}
        for f in self.comfy.output_files(entry):
            name = by_node.get(f.node_id)
            if name is None or f.type != "output":
                continue
            outputs[name].append(remote_join(self.cfg.data_root, *f.rel_path.split("/")))
        for name, paths in outputs.items():
            if not paths:
                raise JobFailed(f"{spec.label}: ComfyUI produced no files for output {name!r}")
            paths.sort()
        return outputs, {"prompt_id": prompt_id, "bypassed": bound.bypassed}

    def _run_task(self, spec: JobSpec) -> tuple[dict[str, list[str]], dict[str, Any]]:
        assert spec.engine.task and spec.engine.image
        args = ["--output-dir", spec.output_dir, "--params", json.dumps(spec.params, ensure_ascii=False)]
        self.progress(f"{spec.label}: running {spec.engine.id} on the workstation")
        data = self.tools.run_task(
            spec.engine.task,
            args,
            gpu=spec.engine.gpu,
            image=spec.engine.image,
            timeout=spec.timeout_s,
            on_line=lambda line: self.progress(f"{spec.label}: {line}") if line.strip() else None,
        )
        outputs = {k: (v if isinstance(v, list) else [v]) for k, v in (data.get("outputs") or {}).items()}
        return outputs, {"result": {k: v for k, v in data.items() if k not in {"outputs", "ok"}}}


def make_runner(app: Any, project: ProjectHandle | None = None) -> JobRunner:
    state = (project or app.project).state
    return JobRunner(app.config, app.executor, app.comfy, state, progress=app.out.progress)
