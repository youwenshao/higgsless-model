"""Licence gate: block engines whose licence doesn't fit the project's usage."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from higgsless.core.manifest import Project, Usage
from higgsless.errors import ValidationError


class Licence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: str
    code: str | None = None
    commercial: bool
    notes: str | None = None


def check_licence(engine_id: str, licence: Licence, project: Project) -> list[str]:
    """Raise if the engine may not be used; return warnings worth surfacing otherwise."""
    warnings: list[str] = []
    if licence.commercial:
        if licence.notes:
            warnings.append(f"{engine_id}: {licence.notes}")
        return warnings
    if project.usage is Usage.COMMERCIAL:
        raise ValidationError(
            f"{engine_id} is licensed {licence.weights} (non-commercial) and this project's usage is commercial",
            hint="pick a different engine for this capability",
        )
    if engine_id not in project.allow_noncommercial:
        raise ValidationError(
            f"{engine_id} is licensed {licence.weights} (non-commercial)",
            hint=f"add {engine_id} to allow_noncommercial in project.yaml to opt in",
        )
    warnings.append(f"{engine_id}: non-commercial licence ({licence.weights}) accepted via allow_noncommercial")
    return warnings
