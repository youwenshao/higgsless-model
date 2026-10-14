"""Error types. Each maps to a stable exit code so agents can branch on failures."""

from __future__ import annotations

from typing import Any


class HiggslessError(Exception):
    exit_code = 1
    kind = "error"

    def __init__(self, message: str, *, hint: str | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "message": self.message}
        if self.hint:
            data["hint"] = self.hint
        if self.details is not None:
            data["details"] = self.details
        return data


class ConfigError(HiggslessError):
    """Missing or invalid configuration, project or manifest."""

    exit_code = 3
    kind = "config"


class RemoteError(HiggslessError):
    """The workstation (SSH or ComfyUI) could not be reached or misbehaved."""

    exit_code = 4
    kind = "remote"


class ValidationError(HiggslessError):
    """Inputs failed validation: workflow bindings, licence gate, lyrics, constraints."""

    exit_code = 5
    kind = "validation"


class JobFailed(HiggslessError):
    """A ComfyUI prompt or tool container ran but failed."""

    exit_code = 6
    kind = "job_failed"


class GateBlocked(HiggslessError):
    """A review gate needs a decision (approve keyframes, pick a take) before continuing."""

    exit_code = 7
    kind = "gate"
