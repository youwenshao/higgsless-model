"""Engine manifest schema (engines/<id>.yaml)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from higgsless.core.licence import Licence

Capability = Literal[
    # video modes
    "swap",
    "rebuild",
    "restyle",
    "composite",
    # images
    "portrait",
    "sheet",
    "keyframe",
    # analysis
    "track",
    "caption",
    "embed",
    # audio
    "separate",
    "transcribe",
    "svs",
    "svc",
    "voice",
    "voice_train",
    "mix",
    # finishing
    "lipsync",
    "upscale",
    "interp",
    "matte",
    "plate",
]


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim_multiple: int = 32
    window: int | None = Field(None, description="Max frames per generation window (4n+1)")
    overlap: int = Field(0, description="Frames shared with the previous window")
    preferred_height: int | None = None
    max_people: int | None = None
    native_fps: float | None = None
    min_face_frac: float | None = Field(None, description="Lip-sync: smallest face height / frame height")


class EngineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    capability: Capability
    backend: Literal["comfyui", "docker"]
    description: str = ""
    default: bool = Field(False, description="Default engine for its capability")
    experimental: bool = False
    workflow: str | None = Field(None, description="Path under workflows/")
    bindings: str | None = Field(None, description="Path under workflows/")
    image: str | None = Field(None, description="Docker image key from the workstation config")
    task: str | None = Field(None, description="Worker task module or container entry point")
    gpu: bool = True
    licence: Licence
    constraints: Constraints = Field(default_factory=Constraints)
    models: list[str] = Field(default_factory=list, description="Model ids from worker/models.lock.yaml")
    node_packs: list[str] = Field(default_factory=list, description="Custom node packs required")
    defaults: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    links: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> EngineSpec:
        if self.backend == "comfyui" and not (self.workflow and self.bindings):
            raise ValueError(f"{self.id}: comfyui engines need workflow and bindings")
        if self.backend == "docker" and not (self.image and self.task):
            raise ValueError(f"{self.id}: docker engines need image and task")
        prefix, _, name = self.id.partition(".")
        if prefix != self.capability or not name:
            raise ValueError(f"{self.id}: engine ids look like <capability>.<name>, e.g. {self.capability}.foo")
        return self

    def params_for(self, profile: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(self.defaults)
        params.update(self.profiles.get(profile, {}))
        params.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return params
