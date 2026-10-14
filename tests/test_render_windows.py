"""Render-stage window chaining against a fake ComfyUI that 'renders' by copying the driving frames."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from higgsless.cli.context import AppContext
from higgsless.cli.output import Output
from higgsless.core.config import load_config
from higgsless.core.project import create_project
from higgsless.pipeline import render as render_stage
from higgsless.pipeline.ingest import ingest
from higgsless.pipeline.shots import extract_shots
from higgsless.pipeline.stage import Stage
from tests.conftest import needs_ffmpeg

pytestmark = [needs_ffmpeg]


class FakeComfy:
    """Implements ComfyClient.run/output_files/free by decoding the input clip into SaveImage files."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root
        self.calls: list[dict] = []

    def free(self) -> None:
        pass

    def run(self, workflow, on_progress=None, timeout_s=None):
        self.calls.append(workflow)
        nodes = {n["_meta"]["title"]: (nid, n) for nid, n in workflow.items()}
        clip = nodes.get("Driving Video") or nodes.get("Source Video")
        src = self.root / clip[1]["inputs"]["file"]
        save_id, save = nodes["Save Frames"]
        prefix = save["inputs"]["filename_prefix"]
        out_dir = self.root / Path(prefix).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "_decode"
        tmp.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), str(tmp / "%05d.png")], check=True)
        frames = sorted(tmp.glob("*.png"))
        length = int((nodes.get("Animate") or nodes.get("SCAIL") or nodes.get("Bernini"))[1]["inputs"]["length"])
        frames += [frames[-1]] * (length - len(frames))  # the real nodes hold the last frame to fill a window
        skip = 0
        if "Animate" in nodes and "Continue Frames" in nodes:
            skip = 1  # WanAnimate2ToVideo trim_image when continuing
        if "SCAIL" in nodes:
            skip = int(nodes["Drop Overlap"][1]["inputs"]["batch_index"])
        names = []
        for i, frame in enumerate(frames[skip:], start=1):
            name = f"{Path(prefix).name}_{i:05d}_.png"
            (out_dir / name).write_bytes(frame.read_bytes())
            names.append({"filename": name, "subfolder": Path(prefix).parent.as_posix(), "type": "output"})
        return "fake", {"outputs": {save_id: {"images": names}}, "status": {"status_str": "success"}}

    @staticmethod
    def output_files(entry):
        from higgsless.remote.comfy import ComfyClient

        return ComfyClient.output_files(entry)


def _video(path: Path, frames: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=128x96:r=24",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            str(path),
        ],
        check=True,
    )


@pytest.mark.parametrize("engine", ["rebuild.wan_animate2", "swap.scail2"])
def test_long_shot_windows_assemble_exact_frames(tmp_path, projects_home, local_config, engine):
    total = 200
    video = tmp_path / "mv.mp4"
    _video(video, total)
    cfg = load_config(local_config)
    app = AppContext(out=Output(json_mode=True))
    app._config = cfg
    fake = FakeComfy(Path(cfg.data_root))
    app._comfy = fake
    look = "new-world" if engine.startswith("rebuild") else "original"
    handle = create_project("win", usage="private", look=look, profile="standard")
    app._project = handle
    stage = Stage(app)
    ingest(stage, video)
    with handle.edit() as data:
        data["characters"]["ava"] = {"prompt": "a singer"}
        data["shots"] = [
            {
                "id": "s001",
                "frames": [0, total],
                "mode": look.replace("new-world", "rebuild").replace("original", "swap"),
                "engine": engine,
                "cast": {"1": "ava"},
            }
        ]
    extract_shots(stage, handle.load().shots)
    # Pretend a keyframe was made and approved.
    keyframe = Path(cfg.data_root) / "projects" / "win" / "shots" / "s001" / "first.png"
    handle.state.write(
        "keyframes/s001", {"path": str(keyframe), "caption": "Character Description: x", "approved": True, "job": "k1"}
    )
    state = render_stage.render_shot(stage, "s001", takes=2)
    assert state["engine"] == engine and set(state["takes"]) == {"1", "2"}
    for take in state["takes"].values():
        frames = json.loads(Path(take["frames_list"]).read_text(encoding="utf-8"))
        assert len(frames) == total == take["count"]
    windows = state["windows"]
    assert sum(w["emit_count"] for w in windows) == total and len(windows) == 3
    # Continuation windows received the previous window's tail.
    renders = [c for c in fake.calls if any(n["_meta"]["title"] == "Save Frames" for n in c.values())]
    titles = [{n["_meta"]["title"] for n in c.values()} for c in renders]
    tail_node = "Continue Frames" if engine == "rebuild.wan_animate2" else "Previous Frames"
    assert [tail_node in t for t in titles] == [False, True, True, False, True, True]
    # Re-running is cached: no new ComfyUI calls.
    before = len(fake.calls)
    render_stage.render_shot(stage, "s001", takes=2)
    assert len(fake.calls) == before
