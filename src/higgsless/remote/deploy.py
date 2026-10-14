"""Copy the worker code (src/higgsless and worker/) to the workstation when it changed."""

from __future__ import annotations

import hashlib
from pathlib import Path

from higgsless.core.paths import data_dir, repo_root
from higgsless.core.project import remote_join
from higgsless.remote.executor import Executor

_SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _skip(rel: Path) -> bool:
    return any(part in _SKIP_PARTS for part in rel.parts) or rel.suffix in {".pyc", ".pyo"}


def _sources() -> list[tuple[Path, str]]:
    package = Path(__file__).resolve().parent.parent
    pairs = [(package, "src/higgsless"), (data_dir("worker"), "worker")]
    root = repo_root()
    if root is not None:
        pairs.append((root / "engines", "engines"))
    return pairs


def tree_hash() -> str:
    digest = hashlib.sha256()
    for local, name in _sources():
        for path in sorted(p for p in local.rglob("*") if p.is_file()):
            rel = path.relative_to(local)
            if _skip(rel):
                continue
            digest.update(f"{name}/{rel.as_posix()}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def deploy_worker(executor: Executor, worker_dir: str, force: bool = False) -> dict[str, object]:
    marker = remote_join(worker_dir, ".deployed")
    current = tree_hash()
    if not force and executor.exists(marker) and executor.read_text(marker).strip() == current:
        return {"deployed": False, "hash": current, "files": 0}
    files = 0
    for local, name in _sources():
        target = remote_join(worker_dir, *name.split("/"))
        executor.remove(target, recursive=True)
        executor.makedirs(target)
        files += executor.put_tree(local, target, skip=_skip)
    executor.write_text(marker, current + "\n")
    return {"deployed": True, "hash": current, "files": files}
