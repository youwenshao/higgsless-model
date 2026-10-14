"""ffmpeg command builders. Pure functions so frame maths can be unit tested."""

from __future__ import annotations

from fractions import Fraction

from higgsless.core.timecode import format_rate

MEZZ_ARGS = ["-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p"]
LOSSLESS_ARGS = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"]


def seek_time(frame: int, rate: Fraction) -> str:
    """Input seek a quarter frame early: accurate seeking keeps frame `frame` and drops `frame - 1`."""
    if frame <= 0:
        return "0"
    return f"{float((Fraction(frame) - Fraction(1, 4)) / rate):.6f}"


def cut_clip(src: str, dst: str, start: int, count: int, rate: Fraction, *, vf: str | None = None,
             codec_args: list[str] | None = None) -> list[str]:
    """Frame-exact clip [start, start+count) re-encoded (no audio)."""
    argv = ["ffmpeg", "-y", "-v", "error", "-ss", seek_time(start, rate), "-i", src, "-frames:v", str(count)]
    if vf:
        argv += ["-vf", vf]
    argv += ["-fps_mode", "passthrough", "-an", *(codec_args or MEZZ_ARGS), dst]
    return argv


def gen_filter(width: int, height: int, decimate: int = 1, start: int | None = None, end: int | None = None) -> str:
    """Decimate (by source frame index), then trim (by generator frame index), then scale."""
    parts = []
    if decimate > 1:
        parts.append(f"select='not(mod(n\\,{decimate}))'")
    if start is not None or end is not None:
        trim = []
        if start is not None:
            trim.append(f"start_frame={start}")
        if end is not None:
            trim.append(f"end_frame={end}")
        parts.append("trim=" + ":".join(trim))
    parts += ["setpts=N/FRAME_RATE/TB", f"scale={width}:{height}:flags=lanczos", "setsar=1"]
    return ",".join(parts)


def encode_frames(pattern: str, dst: str, rate: Fraction, *, crf: int = 16, codec: str = "h264",
                  start_number: int = 1, vf: str | None = None) -> list[str]:
    """PNG sequence -> video at an exact rate."""
    vcodec = {"h264": "libx264", "hevc": "libx265"}[codec]
    argv = ["ffmpeg", "-y", "-v", "error", "-framerate", format_rate(rate), "-start_number", str(start_number),
            "-i", pattern]
    if vf:
        argv += ["-vf", vf]
    argv += ["-c:v", vcodec, "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]
    if codec == "hevc":
        argv += ["-tag:v", "hvc1"]
    argv += ["-r", format_rate(rate), dst]
    return argv


def audio_wav(src: str, dst: str, rate: int = 48000) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "2", "-ar", str(rate), "-c:a", "pcm_s24le", dst]


def proxy(src: str, dst: str, short_edge: int = 480) -> list[str]:
    """Frame-preserving low-res proxy for detection and previews."""
    vf = f"scale=-2:{short_edge}:flags=bilinear"
    return ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf, "-fps_mode", "passthrough", "-c:v", "libx264",
            "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", dst]


def frame_png(src: str, dst: str, frame: int, rate: Fraction) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-ss", seek_time(frame, rate), "-i", src, "-frames:v", "1", dst]
