"""End-to-end check against a reachable ComfyUI (e.g. a CPU validator in WSL) with a shared data root.

ComfyUI must run with --input-directory and --output-directory pointing at the same folder as
--data-root (as seen from ComfyUI's machine). Worker tasks run natively here.

  python scripts/dev/e2e_local.py --comfy-url http://127.0.0.1:8189 --data-root C:/.../e2e/data \
      --steps init,shots,analyze
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def make_short_video(path: Path, frames: int) -> None:
    """One shot of `frames` frames (for slow CPU validators)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=24000/1001",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:r=48000",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def make_video(path: Path) -> None:
    size, rate = "320x240", "24000/1001"
    graph = (
        "[0:v]trim=end_frame=24,setpts=PTS-STARTPTS[a];[1:v]trim=end_frame=36,setpts=PTS-STARTPTS[b];"
        "[2:v]noise=alls=30:allf=t,trim=end_frame=30,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"smptebars=s={size}:r={rate}",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s={size}:r={rate}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x228833:s={size}:r={rate}",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:r=48000",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-frames:v",
            "90",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def cli(env: dict[str, str], *args: str) -> dict:
    print("$ higgsless", " ".join(args), flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "higgsless", "--json", "-v", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=REPO,
        check=False,
    )
    if proc.stderr.strip():
        print(proc.stderr.strip()[-1500:], flush=True)
    data = json.loads(proc.stdout)
    print(json.dumps(data, indent=1, ensure_ascii=False)[:2500], flush=True)
    if not data["ok"]:
        raise SystemExit(f"step failed: {' '.join(args)}")
    return data["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-url", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work", default=str(Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "higgsless" / "e2e"))
    parser.add_argument("--steps", default="init,shots,analyze")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--short", type=int, default=0, help="Use a single-shot clip of N frames")
    args = parser.parse_args()
    work = Path(args.work)
    if args.fresh and (work / "projects").exists():
        shutil.rmtree(work / "projects")
    (work / "projects").mkdir(parents=True, exist_ok=True)
    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    cfg = work / "workstation.yaml"
    cfg.write_text(
        "\n".join(
            [
                "executor: local",
                f"comfy: {{url: '{args.comfy_url}', poll_s: 1.0}}",
                f"data_root: '{Path(args.data_root).as_posix()}'",
                f"worker_dir: '{REPO.as_posix()}'",
                f"hf_cache: '{(work / 'hf').as_posix()}'",
                "tools_mode: native",
                f"native_python: '{Path(sys.executable).as_posix()}'",
            ]
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HIGGSLESS_CONFIG": str(cfg),
        "HIGGSLESS_PROJECTS": str(work / "projects"),
        "PYTHONIOENCODING": "utf-8",
    }
    steps = args.steps.split(",")
    if "init" in steps:
        video = work / "mv.mp4"
        make_short_video(video, args.short) if args.short else make_video(video)
        cli(env, "init", "e2e", "--source", str(video), "--look", "original", "--exist-ok")
    if "shots" in steps:
        cli(env, "-p", "e2e", "shots", "detect", "--force")
    if "analyze" in steps:
        cli(env, "-p", "e2e", "analyze", "--no-captions", "--force")
        cli(env, "-p", "e2e", "shots", "suggest")
    print("e2e ok", flush=True)


if __name__ == "__main__":
    main()
