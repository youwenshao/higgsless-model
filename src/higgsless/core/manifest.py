"""Schema for projects/<name>/project.yaml: the user's (or agent's) intent for a project."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Usage(StrEnum):
    PRIVATE = "private"
    PUBLIC_NONCOMMERCIAL = "public_noncommercial"
    COMMERCIAL = "commercial"


class Look(StrEnum):
    ORIGINAL = "original"  # keep sets, swap performers
    NEW_WORLD = "new-world"  # rebuild every shot around the new characters
    RESTYLE = "restyle"  # restyle the whole video


class Profile(StrEnum):
    DRAFT = "draft"
    STANDARD = "standard"
    MAX = "max"


class Mode(StrEnum):
    AUTO = "auto"
    SWAP = "swap"
    REBUILD = "rebuild"
    RESTYLE = "restyle"
    COMPOSITE = "composite"
    PASSTHROUGH = "passthrough"


class Switch(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class VoiceKind(StrEnum):
    REFERENCE = "reference"
    DESIGNED = "designed"
    RVC = "rvc"


class Consent(StrEnum):
    SELF = "self"
    WRITTEN = "written"
    SYNTHETIC = "synthetic"


LANGUAGES = ("en", "zh", "yue", "ja", "ko")
PROFILE_RESOLUTION = {Profile.DRAFT: "480p", Profile.STANDARD: "720p", Profile.MAX: "720p"}
_TRANSITION = re.compile(r"^(cut|dissolve:\d+|fade_in:\d+|fade_out:\d+)$")
_SHOT_ID = re.compile(r"^s\d{3,4}[a-z]?$")


class Source(_Strict):
    path: str = Field(description="Path relative to the remote project root")
    original_name: str
    rate: str = Field(description="Exact frame rate, e.g. 24000/1001")
    frames: int
    width: int
    height: int
    duration_s: float
    has_audio: bool
    audio_rate: int | None = None
    sha256: str | None = None


class Character(_Strict):
    prompt: str = ""
    images: list[str] = Field(default_factory=list, description="Reference images (remote project paths)")
    style: str | None = None
    stylised: bool = Field(False, description="Non-photoreal; routes lip-sync to InfiniteTalk")
    caption_zh: str | None = Field(None, description="Wan-Animate-2 appearance caption, filled by analyze")
    sheet: str | None = Field(None, description="Character sheet, set by `char create`")


class Voice(_Strict):
    kind: VoiceKind
    consent: Consent
    reference: str | None = None
    describe: str | None = None
    rvc_model: str | None = None
    rvc_index: str | None = None
    language: str = "en"
    pitch_shift: int = 0

    @model_validator(mode="after")
    def _check(self) -> Voice:
        if self.kind is VoiceKind.REFERENCE and not self.reference:
            raise ValueError("reference voices need `reference` (a clean 10-30 s clip)")
        if self.kind is VoiceKind.DESIGNED:
            if not self.describe:
                raise ValueError("designed voices need `describe` (a text description of the voice)")
            if self.consent is not Consent.SYNTHETIC:
                raise ValueError("designed voices belong to no real person: use consent: synthetic")
        if self.kind is VoiceKind.RVC and not self.rvc_model:
            raise ValueError("rvc voices need `rvc_model` (trained with `voice train`)")
        if self.kind is not VoiceKind.DESIGNED and self.consent is Consent.SYNTHETIC:
            raise ValueError("voices cloned from a recording need consent: self or written")
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        return self


class Shot(_Strict):
    id: str
    frames: list[int] = Field(description="[start, end) in source frames")
    mode: Mode = Mode.AUTO
    engine: str | None = None
    cast: dict[str, str] = Field(default_factory=dict, description="Track or performer id -> character")
    lipsync: Switch = Switch.AUTO
    prompt: str | None = None
    takes: int | None = None
    transition: str = "cut"
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _SHOT_ID.match(v):
            raise ValueError("shot ids look like s001, s012 or s012b")
        return v

    @field_validator("frames")
    @classmethod
    def _frames(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or v[0] < 0 or v[1] <= v[0]:
            raise ValueError("frames must be [start, end) with end > start >= 0")
        return v

    @field_validator("transition")
    @classmethod
    def _transition(cls, v: str) -> str:
        if not _TRANSITION.match(v):
            raise ValueError("transition must be cut, dissolve:<frames>, fade_in:<frames> or fade_out:<frames>")
        return v

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[1]

    @property
    def length(self) -> int:
        return self.frames[1] - self.frames[0]


class Delivery(_Strict):
    resolution: Literal["source", "720p", "1080p", "2160p"] = "1080p"
    codec: Literal["h264", "hevc"] = "h264"
    crf: int = 18
    upscaler: str | None = None


class AudioConfig(_Strict):
    language: str = "en"
    lead_voice: str | None = None
    backing: Literal["keep", "convert", "mute"] = "keep"
    lyrics_new: str = "lyrics_new.yaml"
    singer: str | None = Field(None, description="Engine override for singing synthesis")

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        if v not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        return v


class Project(_Strict):
    version: int = 1
    project: str
    usage: Usage = Usage.PUBLIC_NONCOMMERCIAL
    look: Look = Look.NEW_WORLD
    style: str | None = Field(None, description="Restyle prompt when look is restyle")
    profile: Profile = Profile.STANDARD
    render_resolution: Literal["480p", "576p", "720p"] | None = None
    takes: int = 1
    delivery: Delivery = Field(default_factory=Delivery)
    source: Source | None = None
    characters: dict[str, Character] = Field(default_factory=dict)
    voices: dict[str, Voice] = Field(default_factory=dict)
    cast: dict[str, str] = Field(default_factory=dict, description="Performer id (P1...) -> character")
    engines: dict[str, str] = Field(default_factory=dict, description="Capability -> engine id override")
    allow_noncommercial: list[str] = Field(default_factory=list)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    shots: list[Shot] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Project:
        ids = [s.id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("shot ids must be unique")
        for prev, cur in zip(self.shots, self.shots[1:], strict=False):
            if cur.start < prev.end:
                raise ValueError(f"shots {prev.id} and {cur.id} overlap or are out of order")
        if self.source is not None and self.shots and self.shots[-1].end > self.source.frames:
            raise ValueError(f"shot {self.shots[-1].id} ends after the source ({self.source.frames} frames)")
        known = set(self.characters)
        for performer, name in self.cast.items():
            if name not in known:
                raise ValueError(f"cast maps {performer} to unknown character {name!r}")
        for shot in self.shots:
            for track, name in shot.cast.items():
                if name not in known:
                    raise ValueError(f"shot {shot.id} casts {track} as unknown character {name!r}")
        if self.audio.lead_voice and self.audio.lead_voice not in self.voices:
            raise ValueError(f"audio.lead_voice {self.audio.lead_voice!r} is not a defined voice")
        if self.look is Look.RESTYLE and not self.style:
            raise ValueError("look: restyle needs `style` (for example 'hand-painted anime, cel shading')")
        return self

    @property
    def resolution(self) -> str:
        return self.render_resolution or PROFILE_RESOLUTION[self.profile]

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(shot_id)
