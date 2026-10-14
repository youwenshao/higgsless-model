"""`higgsless doctor`: check that the workstation is ready."""

from __future__ import annotations

import os
from typing import Any

import typer

from higgsless.cli.context import AppContext, get_ctx
from higgsless.cli.output import table
from higgsless.core.models import ModelsLock, check_remote, models_for
from higgsless.engines.registry import registry
from higgsless.engines.runner import load_workflow
from higgsless.errors import ConfigError, HiggslessError, ValidationError
from higgsless.remote.deploy import tree_hash


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = None, hint: str | None = None, level: str = "error") -> bool:
        self.items.append({"name": name, "ok": ok, "level": "ok" if ok else level, "detail": detail, "hint": hint})
        return ok

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.items if not c["ok"] and c["level"] == "error"]


def _selected_engines(app_ctx: AppContext) -> list[Any]:
    reg = registry()
    try:
        project = app_ctx.project.load()
    except ConfigError:
        project = None
    capabilities = sorted({e.capability for e in reg.engines.values()})
    return [reg.default_for(cap, project) for cap in capabilities]


def _unix_checks(app_ctx: AppContext, checks: Checks) -> None:
    ex, cfg = app_ctx.executor, app_ctx.config
    res = ex.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], check=False, timeout=30
    )
    if res.ok and res.stdout.strip():
        name, driver, mem = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
        major = int(driver.split(".")[0])
        checks.add(
            "gpu",
            major >= 580,
            f"{name}, driver {driver}, {mem}",
            hint=None if major >= 580 else "install NVIDIA driver r580+ for CUDA 13 (cu130)",
        )
    else:
        checks.add("gpu", False, res.stderr.strip()[:200] or "nvidia-smi failed", hint="install the NVIDIA driver")
    res = ex.run(
        ["bash", "-lc", f"free -g | awk '/Mem:/ {{print $2}}'; df -BG --output=avail {cfg.data_root} | tail -1"],
        check=False,
        timeout=30,
    )
    if res.ok:
        lines = res.stdout.split()
        ram_gb, free_gb = int(lines[0]), int(lines[1].rstrip("G"))
        checks.add("ram", ram_gb >= 60, f"{ram_gb} GB", hint="64 GB+ recommended; add swap", level="warning")
        checks.add(
            "disk",
            free_gb >= 300,
            f"{free_gb} GB free under {cfg.data_root}",
            hint="models and renders need several hundred GB",
            level="warning",
        )
    res = ex.run(["docker", "version", "--format", "{{.Server.Version}}"], check=False, timeout=30)
    checks.add(
        "docker",
        res.ok,
        res.stdout.strip() or res.stderr.strip()[:200],
        hint="run worker/bootstrap.sh (STEPS=docker) and log in again for the docker group",
    )
    if res.ok:
        missing = []
        for key, tag in cfg.docker.images.items():
            if not ex.run(["docker", "image", "inspect", tag], check=False, timeout=30).ok:
                missing.append(key)
        checks.add(
            "tool_images",
            not missing,
            {"missing": missing} if missing else "all present",
            hint="higgsless worker build",
            level="warning",
        )


def doctor(
    ctx: typer.Context,
    profile: str = typer.Option("standard", help="Profile whose model files must be present."),
) -> None:
    """Check SSH, GPU, ComfyUI (torch build, node packs), models, Docker images and worker code."""
    app_ctx = get_ctx(ctx)
    checks = Checks()
    cfg = app_ctx.config
    checks.add("config", True, {"executor": cfg.executor, "comfy": cfg.comfy.url, "data_root": cfg.data_root})

    remote_ok = False
    try:
        res = app_ctx.executor.run(
            ["uname", "-srm"] if os.name != "nt" or cfg.executor == "ssh" else ["cmd", "/c", "ver"], timeout=30
        )
        remote_ok = checks.add("executor", True, f"{cfg.executor}: {res.stdout.strip()}")
    except HiggslessError as exc:
        checks.add("executor", False, exc.message, hint=exc.hint)

    if remote_ok and (cfg.executor == "ssh" or os.name != "nt"):
        _unix_checks(app_ctx, checks)
        marker = f"{cfg.worker_dir.rstrip('/')}/.deployed"
        deployed = app_ctx.executor.read_text(marker).strip() if app_ctx.executor.exists(marker) else None
        checks.add(
            "worker_code",
            deployed == tree_hash(),
            deployed or "not deployed",
            hint="higgsless worker deploy",
            level="warning",
        )

    engines = _selected_engines(app_ctx)
    try:
        stats = app_ctx.comfy.system_stats()
        system = stats.get("system", {})
        torch_version = str(system.get("pytorch_version", ""))
        devices = [f"{d.get('name')} ({round(d.get('vram_total', 0) / 2**30)} GB)" for d in stats.get("devices", [])]
        checks.add(
            "comfyui", True, {"version": system.get("comfyui_version"), "torch": torch_version, "devices": devices}
        )
        checks.add(
            "torch_cu130",
            "cu13" in torch_version,
            torch_version,
            hint="cu128 measured ~2x slower for Wan-Animate-2; reinstall with TORCH_INDEX=.../cu130",
            level="warning",
        )
        info = app_ctx.comfy.object_info()
        missing: dict[str, list[str]] = {}
        for engine in engines:
            if engine.backend != "comfyui":
                continue
            workflow, _ = load_workflow(engine)
            absent = sorted({n["class_type"] for n in workflow.values() if n["class_type"] not in info})
            if absent:
                missing[engine.id] = absent
        hints = sorted({p for e in engines if e.id in missing for p in e.node_packs})
        checks.add(
            "node_classes",
            not missing,
            missing or "all present",
            hint=f"install node packs: {hints} (worker/custom_nodes.txt)" if missing else None,
        )
    except HiggslessError as exc:
        checks.add("comfyui", False, exc.message, hint=exc.hint)

    if remote_ok:
        lock = ModelsLock.load()
        ids: list[str] = []
        for engine in engines:
            ids += [m for m in models_for(engine.models, profile) if m not in ids]
        missing_models = []
        for model_id in ids:
            status = check_remote(app_ctx.executor, cfg, lock.get(model_id))
            if status["status"] != "present":
                missing_models.append({"id": model_id, **status})
        checks.add(
            "models",
            not missing_models,
            missing_models or f"{len(ids)} files present",
            hint=f"higgsless models sync --profile {profile}",
        )

    rows = [[c["name"], c["level"], str(c["detail"])[:80]] for c in checks.items]
    if checks.failed:
        raise ValidationError(
            f"{len(checks.failed)} check(s) failed",
            details={"checks": checks.items},
            hint="; ".join(c["hint"] for c in checks.failed if c["hint"]),
        )
    app_ctx.out.result({"ok": True, "checks": checks.items}, table("doctor", ["check", "status", "detail"], rows))
