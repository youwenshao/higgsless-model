"""Exact frame and time maths. Timing is always integer frames at a rational frame rate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

# Common broadcast rates written as decimals by tools and humans.
_NTSC = {
    "23.976": Fraction(24000, 1001),
    "23.98": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "47.952": Fraction(48000, 1001),
    "59.94": Fraction(60000, 1001),
    "119.88": Fraction(120000, 1001),
}


def parse_rate(rate: str | int | float | Fraction) -> Fraction:
    """Parse "24000/1001", "23.976", "25" or a number into an exact rate."""
    if isinstance(rate, Fraction):
        value = rate
    elif isinstance(rate, int):
        value = Fraction(rate)
    elif isinstance(rate, float):
        value = _from_decimal(repr(rate))
    else:
        text = rate.strip()
        if "/" in text:
            num, den = text.split("/", 1)
            value = Fraction(int(num), int(den))
        else:
            value = _from_decimal(text)
    if value <= 0:
        raise ValueError(f"frame rate must be positive: {rate!r}")
    return value


def _from_decimal(text: str) -> Fraction:
    for key, exact in _NTSC.items():
        if abs(float(text) - float(key)) < 0.0015:
            return exact
    return Fraction(text).limit_denominator(1001)


def format_rate(rate: Fraction) -> str:
    return str(rate.numerator) if rate.denominator == 1 else f"{rate.numerator}/{rate.denominator}"


def frames_to_seconds(frames: int, rate: Fraction) -> Fraction:
    return Fraction(frames) / rate


def seconds_to_frame(seconds: float | Fraction, rate: Fraction) -> int:
    """Index of the frame displayed at `seconds` (floor, with a tolerance for float noise)."""
    exact = Fraction(seconds).limit_denominator(1_000_000) * rate
    return math.floor(exact + Fraction(1, 10_000))


def frame_to_sample(frame: int, rate: Fraction, sample_rate: int) -> int:
    """First audio sample belonging to `frame`. Contiguous frames give contiguous sample ranges."""
    return math.floor(Fraction(frame) * sample_rate / rate)


def frame_to_timecode(frame: int, rate: Fraction) -> str:
    """HH:MM:SS.mmm for humans (not SMPTE drop-frame)."""
    total_ms = math.floor(frames_to_seconds(frame, rate) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


@dataclass(frozen=True)
class FramePolicy:
    """How source frames map to generator frames.

    Sources at or below `max_gen_fps` map one-to-one. Faster sources are decimated by an integer
    factor and interpolated back up by the same factor after rendering, so timing stays exact.
    """

    source_rate: Fraction
    decimate: int

    @property
    def gen_rate(self) -> Fraction:
        return self.source_rate / self.decimate

    def gen_frames(self, source_frames: int) -> int:
        return math.ceil(source_frames / self.decimate)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_rate": format_rate(self.source_rate),
            "decimate": self.decimate,
            "gen_rate": format_rate(self.gen_rate),
        }


def plan_frame_policy(rate: Fraction, max_gen_fps: int = 30) -> FramePolicy:
    if rate <= max_gen_fps:
        return FramePolicy(rate, 1)
    return FramePolicy(rate, math.ceil(rate / max_gen_fps))
