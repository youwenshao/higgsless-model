"""Ingest: upload the source, probe exact timing, make the proxy and the original audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from higgsless.core.timecode import parse_rate, plan_frame_policy
from higgsless.errors import ConfigError
from higgsless.pipeline.stage import Stage


def ingest(stage: Stage, source: Path, *, force: bool = False) -> dict[str, Any]:
    if not source.exists():
        raise ConfigError(f"source video not found: {source}")
    remote_dir = stage.rpath("source")
    stage.ex.makedirs(remote_dir)
    remote = stage.join(remote_dir, "source" + source.suffix.lower())
    local_size = source.stat().st_size
    st = stage.ex.stat(remote)
    if force or st is None or st.size != local_size:
        stage.progress(f"uploading {source.name} ({local_size / 1e6:.0f} MB)")
        stage.ex.put(source, remote)

    probe = stage.task("probe", {"path": remote, "hash": True}, remote_dir)["probe"]
    for warning in probe.get("warnings", []):
        stage.app.out.warn(warning)
    stage.progress("making the proxy and extracting audio")
    derived = stage.task(
        "extract",
        {"action": "ingest", "source": remote, "frames": probe["frames"], "has_audio": probe["has_audio"]},
        remote_dir,
    )["outputs"]

    with stage.handle.edit() as data:
        data["source"] = {
            "path": f"source/source{source.suffix.lower()}",
            "original_name": source.name,
            "rate": probe["rate"],
            "frames": probe["frames"],
            "width": probe["width"],
            "height": probe["height"],
            "duration_s": probe["duration_s"],
            "has_audio": probe["has_audio"],
            "audio_rate": probe["audio_rate"],
            "sha256": probe.get("sha256"),
        }
    policy = plan_frame_policy(parse_rate(probe["rate"]))
    state = {
        "remote_source": remote,
        "probe": probe,
        "proxy": derived["proxy"][0],
        "audio": (derived.get("audio") or [None])[0],
        "frame_policy": policy.to_dict(),
    }
    stage.handle.state.write("ingest", state)
    return state
