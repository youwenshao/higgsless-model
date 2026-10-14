"""worker/models.lock.yaml: model files per engine, and syncing them onto the workstation."""

from __future__ import annotations

import posixpath
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from higgsless.core.config import WorkstationConfig
from higgsless.core.paths import data_dir
from higgsless.errors import ConfigError, ValidationError


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    file: str = Field(description="Path inside the repo, or '*' for a whole-repo snapshot into the HF cache")
    dest: str = Field(description="ComfyUI models subfolder, or 'hf' for the shared Hugging Face cache")
    name: str | None = Field(None, description="Final filename (default: basename of file)")
    revision: str = "main"
    size: int | None = None
    sha256: str | None = None
    licence: str | None = None
    gated: bool = False
    note: str | None = None

    @property
    def filename(self) -> str:
        return self.name or posixpath.basename(self.file)

    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


class ModelsLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelFile] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> ModelsLock:
        path = path or (data_dir("worker") / "models.lock.yaml")
        if not path.exists():
            raise ConfigError(f"missing {path}")
        return cls.model_validate(YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {})

    def get(self, model_id: str) -> ModelFile:
        try:
            return self.models[model_id]
        except KeyError:
            raise ConfigError(f"model {model_id!r} is not in worker/models.lock.yaml") from None


def split_profile(ref: str) -> tuple[str, str | None]:
    """Model refs may be pinned to a profile: `wan_animate2_bf16@max`."""
    model_id, _, profile = ref.partition("@")
    return model_id, (profile or None)


def models_for(engine_models: list[str], profile: str) -> list[str]:
    selected = []
    for ref in engine_models:
        model_id, only = split_profile(ref)
        if only is None or only == profile:
            selected.append(model_id)
    return selected


def remote_path(cfg: WorkstationConfig, model: ModelFile) -> str:
    if model.dest == "hf":
        return posixpath.join(cfg.hf_cache, "hub", "models--" + model.repo.replace("/", "--"))
    return posixpath.join(cfg.models_root, model.dest, model.filename)


def sync_script(cfg: WorkstationConfig, model: ModelFile, verify: bool) -> str:
    """Bash script that downloads one model file (resumable) and optionally verifies sha256."""
    if model.dest == "hf":
        image = cfg.docker.images["tools"]
        pattern = "" if model.file == "*" else f" --include {shlex.quote(model.file)}"
        return (
            "set -euo pipefail\n"
            f"mkdir -p {shlex.quote(cfg.hf_cache)}\n"
            f"docker run --rm -v {shlex.quote(cfg.hf_cache)}:/hf -e HF_HOME=/hf "
            '${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} '
            f"{shlex.quote(image)} hf download {shlex.quote(model.repo)}{pattern} "
            f"--revision {shlex.quote(model.revision)}\n"
        )
    final = remote_path(cfg, model)
    lines = [
        "set -euo pipefail",
        f"final={shlex.quote(final)}",
        'mkdir -p "$(dirname "$final")"',
        'if [ ! -s "$final" ]; then',
        '  auth=(); if [ -n "${HF_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi',
        f'  curl -L --fail --retry 5 --retry-delay 5 -C - "${{auth[@]}}" -o "$final.part" {shlex.quote(model.url())}',
        '  mv "$final.part" "$final"',
        "fi",
    ]
    if model.size:
        lines.append(
            f'size=$(stat -c %s "$final"); [ "$size" -eq {model.size} ] || '
            f'{{ echo "size mismatch: $size != {model.size}" >&2; exit 3; }}'
        )
    if verify and model.sha256:
        lines.append(f'echo "{model.sha256}  $final" | sha256sum -c --quiet -')
    return "\n".join(lines) + "\n"


def check_remote(executor: Any, cfg: WorkstationConfig, model: ModelFile) -> dict[str, Any]:
    path = remote_path(cfg, model)
    st = executor.stat(path)
    if st is None:
        return {"path": path, "status": "missing"}
    if model.dest != "hf" and model.size and st.size != model.size:
        return {"path": path, "status": "size_mismatch", "size": st.size, "expected": model.size}
    return {"path": path, "status": "present", "size": st.size}


def sync_models(
    executor: Any, cfg: WorkstationConfig, lock: ModelsLock, ids: list[str], *, verify: bool, progress: Any = None
) -> list[dict[str, Any]]:
    results = []
    for model_id in ids:
        model = lock.get(model_id)
        before = check_remote(executor, cfg, model)
        if before["status"] == "present" and not verify:
            results.append({"id": model_id, **before, "action": "skipped"})
            continue
        if progress:
            progress(f"downloading {model_id} ({model.repo}/{model.file})")
        try:
            executor.run(["bash", "-lc", sync_script(cfg, model, verify)], timeout=6 * 3600)
        except Exception as exc:  # noqa: BLE001 - report per model and continue
            results.append({"id": model_id, "status": "failed", "error": str(exc)})
            continue
        results.append({"id": model_id, **check_remote(executor, cfg, model), "action": "downloaded"})
    failed = [r["id"] for r in results if r.get("status") != "present"]
    if failed:
        raise ValidationError(f"{len(failed)} model(s) failed to sync: {failed}", details=results)
    return results
