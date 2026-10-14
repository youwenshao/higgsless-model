"""Workstation connection config (config/workstation.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML

from higgsless.core.paths import config_dir
from higgsless.errors import ConfigError

DEFAULT_IMAGES = {
    "tools": "higgsless/tools:latest",
    "soulx": "higgsless/soulx:latest",
    "applio": "higgsless/applio:latest",
    "voxcpm": "higgsless/voxcpm:latest",
    "acestep": "higgsless/acestep:latest",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SshConfig(_Strict):
    host: str = Field(description="Hostname, Tailscale name or ~/.ssh/config alias")
    user: str | None = None
    port: int = 22
    key_path: str | None = None
    known_hosts: Literal["default", "ignore"] = "default"
    connect_timeout_s: float = 15.0


class ComfyConfig(_Strict):
    url: str = "http://127.0.0.1:8188"
    timeout_s: float = 6 * 3600
    poll_s: float = 2.0


class DockerConfig(_Strict):
    gpus: str = "all"
    images: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_IMAGES))
    user: str | None = Field(None, description="uid:gid for containers; queried over SSH when unset")
    extra_args: list[str] = Field(default_factory=list)


class WorkstationConfig(_Strict):
    executor: Literal["ssh", "local"] = "ssh"
    ssh: SshConfig | None = None
    comfy: ComfyConfig = Field(default_factory=ComfyConfig)
    data_root: str = "/srv/higgsless"
    comfy_root: str = "/opt/comfyui"
    comfy_python: str = "/opt/comfyui/.venv/bin/python"
    hf_cache: str = "/srv/higgsless/_hf"
    worker_dir: str = "/srv/higgsless/_worker"
    tools_mode: Literal["docker", "native"] = "docker"
    native_python: str = "python3"
    docker: DockerConfig = Field(default_factory=DockerConfig)

    @property
    def models_root(self) -> str:
        return f"{self.comfy_root.rstrip('/')}/models"


def _candidates(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    paths = [config_dir() / "workstation.yaml", Path.home() / ".config" / "higgsless" / "workstation.yaml"]
    return paths


def load_config(path: Path | None = None) -> WorkstationConfig:
    found = next((p for p in _candidates(path) if p.exists()), None)
    data: dict = {}
    if found is not None:
        data = YAML(typ="safe").load(found.read_text(encoding="utf-8")) or {}
    elif path is not None:
        raise ConfigError(f"config file not found: {path}")
    elif not _env_overrides_present():
        raise ConfigError(
            "no workstation config found",
            hint="copy config/workstation.example.yaml to config/workstation.yaml and set ssh.host and comfy.url",
        )
    _apply_env(data)
    try:
        cfg = WorkstationConfig.model_validate(data)
    except PydanticValidationError as exc:
        raise ConfigError(f"invalid workstation config ({found})", details=exc.errors(include_url=False)) from exc
    if cfg.executor == "ssh" and cfg.ssh is None:
        raise ConfigError("executor is 'ssh' but no ssh section is configured", hint="set ssh.host")
    return cfg


def _env_overrides_present() -> bool:
    return any(k in os.environ for k in ("HIGGSLESS_SSH_HOST", "HIGGSLESS_EXECUTOR", "HIGGSLESS_COMFY_URL"))


def _apply_env(data: dict) -> None:
    if host := os.environ.get("HIGGSLESS_SSH_HOST"):
        data.setdefault("ssh", {})["host"] = host
    if url := os.environ.get("HIGGSLESS_COMFY_URL"):
        data.setdefault("comfy", {})["url"] = url
    if root := os.environ.get("HIGGSLESS_DATA_ROOT"):
        data["data_root"] = root
    if executor := os.environ.get("HIGGSLESS_EXECUTOR"):
        data["executor"] = executor
