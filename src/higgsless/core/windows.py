"""Render window planning and resolution snapping for Wan-family video models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

RESOLUTION_PRESETS = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}


def length_4n1(frames: int) -> int:
    """Smallest 4n+1 frame count that covers `frames`."""
    if frames <= 1:
        return 1
    return 4 * math.ceil((frames - 1) / 4) + 1


def snap(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def gen_size(src_w: int, src_h: int, short_edge: int, multiple: int = 32) -> tuple[int, int]:
    """Scale so the short edge is about `short_edge`, snapping each side to `multiple`.

    The aspect error from snapping is under ~3% and is undone at conform, so nothing is cropped.
    """
    scale = short_edge / min(src_w, src_h)
    return snap(src_w * scale, multiple), snap(src_h * scale, multiple)


def resolve_short_edge(resolution: str | int) -> int:
    if isinstance(resolution, int):
        return resolution
    try:
        return RESOLUTION_PRESETS[resolution]
    except KeyError as exc:
        raise ValueError(f"unknown resolution {resolution!r}; use one of {sorted(RESOLUTION_PRESETS)}") from exc


@dataclass(frozen=True)
class Window:
    index: int
    start: int  # first frame (shot-relative, generator frames) fed to the model
    length: int  # 4n+1 frames generated, including padding
    overlap: int  # leading frames conditioned on the previous window (dropped from output)
    pad: int  # trailing frames repeated past the end of the shot (dropped from output)

    @property
    def emit_start(self) -> int:
        return self.start + self.overlap

    @property
    def emit_count(self) -> int:
        return self.length - self.overlap - self.pad

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data.update(emit_start=self.emit_start, emit_count=self.emit_count)
        return data


def plan_windows(total: int, window: int = 81, overlap: int = 1) -> list[Window]:
    """Split `total` frames into 4n+1 windows that overlap by `overlap` frames.

    Window k>0 starts `overlap` frames before the previous window ended; those frames come from the
    previous window's output and are dropped. The last window shrinks to the smallest 4n+1 length
    that covers what is left, and any remainder is padding.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if window < 5 or (window - 1) % 4:
        raise ValueError(f"window must be 4n+1 and at least 5, got {window}")
    if not 0 <= overlap < window - 1:
        raise ValueError(f"overlap must be in [0, {window - 2}]")

    windows: list[Window] = []
    emitted = 0
    index = 0
    while emitted < total:
        lead = overlap if index else 0
        start = emitted - lead
        needed = (total - emitted) + lead
        length = min(window, length_4n1(needed))
        pad = max(0, start + length - total)
        win = Window(index=index, start=start, length=length, overlap=lead, pad=pad)
        windows.append(win)
        emitted += win.emit_count
        index += 1
    return windows
