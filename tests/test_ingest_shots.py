import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO, needs_ffmpeg

pytestmark = [needs_ffmpeg, pytest.mark.ffmpeg]


def make_video(path: Path) -> None:
    """90 frames at 23.976 fps with hard cuts at 24 and 60, plus a sine tone."""
    size, rate = "320x240", "24000/1001"
    graph = (
        "[0:v]trim=end_frame=24,setpts=PTS-STARTPTS[a];"
        "[1:v]trim=end_frame=36,setpts=PTS-STARTPTS[b];"
        "[2:v]noise=alls=30:allf=t,trim=end_frame=30,setpts=PTS-STARTPTS[c];"
        "[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"smptebars=s={size}:r={rate}",
         "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}",
         "-f", "lavfi", "-i", f"color=c=0x228833:s={size}:r={rate}",
         "-f", "lavfi", "-i", "sine=f=440:r=48000",
         "-filter_complex", graph, "-map", "[v]", "-map", "3:a",
         "-frames:v", "90", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


def cli(*args: str) -> dict:
    proc = subprocess.run([sys.executable, "-m", "higgsless", "--json", *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO, env=os.environ.copy(), check=False)
    assert proc.stdout.strip(), proc.stderr
    data = json.loads(proc.stdout)
    assert data["ok"], json.dumps(data, indent=1) + proc.stderr
    return data["data"]


def test_init_ingest_detect(tmp_path, projects_home, local_config):
    video = tmp_path / "mv.mp4"
    make_video(video)
    data = cli("init", "demo", "--source", str(video), "--look", "original")
    probe = data["ingest"]["probe"]
    assert probe["rate"] == "24000/1001" and probe["frames"] == 90 and probe["has_audio"]

    summary = cli("-p", "demo", "shots", "detect")
    shots = cli("-p", "demo", "shots", "list")
    assert [s["frames"] for s in shots] == [[0, 24], [24, 60], [60, 90]], shots
    assert Path(summary["contact_sheet"]).exists()

    data_root = Path(local_config.read_text().split("data_root: '")[1].split("'")[0])
    mezz = data_root / "projects" / "demo" / "shots" / "s002" / "mezz.mp4"
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(mezz)], capture_output=True, text=True)
    assert out.stdout.strip() == "36"

    split = cli("-p", "demo", "shots", "split", "s002", "--at", "40")
    assert split["shots"] == ["s002", "s002b"]
    merged = cli("-p", "demo", "shots", "merge", "s002", "s002b")
    assert merged["shot"] == "s002"
    assert [s["frames"] for s in cli("-p", "demo", "shots", "list")][1] == [24, 60]
