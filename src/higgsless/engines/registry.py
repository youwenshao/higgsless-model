"""Load engine manifests and pick engines per capability."""

from __future__ import annotations

import difflib
from functools import cache
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError
from ruamel.yaml import YAML

from higgsless.core.licence import check_licence
from higgsless.core.manifest import Project
from higgsless.core.paths import data_dir
from higgsless.engines.spec import EngineSpec
from higgsless.errors import ConfigError


class Registry:
    def __init__(self, engines: dict[str, EngineSpec]) -> None:
        self.engines = engines

    @classmethod
    def load(cls, directory: Path | None = None) -> Registry:
        directory = directory or data_dir("engines")
        yaml = YAML(typ="safe")
        engines: dict[str, EngineSpec] = {}
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.load(path.read_text(encoding="utf-8")) or {}
            try:
                spec = EngineSpec.model_validate(raw)
            except PydanticValidationError as exc:
                raise ConfigError(
                    f"invalid engine manifest {path.name}", details=exc.errors(include_url=False)
                ) from exc
            if spec.id != path.stem:
                raise ConfigError(f"engine file {path.name} declares id {spec.id!r}; they must match")
            engines[spec.id] = spec
        return cls(engines)

    def get(self, engine_id: str) -> EngineSpec:
        try:
            return self.engines[engine_id]
        except KeyError:
            close = difflib.get_close_matches(engine_id, list(self.engines), n=3)
            raise ConfigError(
                f"unknown engine {engine_id!r}", hint=f"did you mean {close}?" if close else None
            ) from None

    def by_capability(self, capability: str) -> list[EngineSpec]:
        return [e for e in self.engines.values() if e.capability == capability]

    def default_for(self, capability: str, project: Project | None = None) -> EngineSpec:
        if project is not None and capability in project.engines:
            return self.get(project.engines[capability])
        candidates = self.by_capability(capability)
        for engine in candidates:
            if engine.default:
                return engine
        if candidates:
            return candidates[0]
        raise ConfigError(f"no engine provides capability {capability!r}")

    def resolve(self, capability: str, project: Project, engine_id: str | None = None) -> tuple[EngineSpec, list[str]]:
        """Pick the engine (explicit, project default, or registry default) and run the licence gate."""
        engine = self.get(engine_id) if engine_id else self.default_for(capability, project)
        if engine.capability != capability:
            raise ConfigError(f"{engine.id} provides {engine.capability!r}, not {capability!r}")
        warnings = check_licence(engine.id, engine.licence, project)
        return engine, warnings


@cache
def registry() -> Registry:
    return Registry.load()
