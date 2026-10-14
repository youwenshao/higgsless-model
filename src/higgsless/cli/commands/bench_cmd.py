"""`higgsless bench`: measure seconds per render window on this workstation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import typer

from higgsless.cli.context import get_ctx
from higgsless.core.paths import config_dir
from higgsless.core.project import remote_join
from higgsless.core.state import StateStore
from higgsless.core.windows import gen_size, length_4n1, resolve_short_edge
from higgsless.engines.registry import registry
from higgsless.engines.runner import JobRunner, JobSpec, load_workflow
from higgsless.remote.deploy import deploy_worker
from higgsless.remote.tools import ToolRunner


def bench(
    ctx: typer.Context,
    engine_id: str = typer.Option("rebuild.wan_animate2", "--engine", "-e"),
    profile: str = typer.Option("draft", help="draft | standard | max"),
    frames: int = typer.Option(81, help="Frames per window (rounded up to 4n+1)."),
    resolution: str | None = typer.Option(None, help="480p | 720p (default from profile)."),
    repeat: int = typer.Option(1, help="Runs to average."),
) -> None:
    """Run one synthetic render window and record its timing in config/bench.json."""
    app_ctx = get_ctx(ctx)
    cfg = app_ctx.config
    engine = registry().get(engine_id)
    short = resolve_short_edge(resolution or ("480p" if profile == "draft" else "720p"))
    width, height = gen_size(16 * 9, 9 * 9, short, engine.constraints.dim_multiple)
    length = length_4n1(frames)
    deploy_worker(app_ctx.executor, cfg.worker_dir)
    synth_dir = remote_join(cfg.data_root, "_bench", "inputs")
    tools = ToolRunner(cfg, app_ctx.executor, app_ctx.comfy)
    media = tools.run_task(
        "synth", ["--output-dir", synth_dir], params={"width": width, "height": height, "frames": length}
    )
    _, bindings = load_workflow(engine)
    rel_ref = media["outputs"]["reference_image"][0][len(cfg.data_root) :].lstrip("/\\").replace("\\", "/")
    candidates = {
        "reference_image": rel_ref,
        "driving_video": media["outputs"]["driving_video"][0],
        "source_video": media["outputs"]["driving_video"][0],
        "width": width,
        "height": height,
        "length": length,
        "prompt": "a person dancing in a studio",
        "seed": 1,
    }
    params = engine.params_for(profile, {k: v for k, v in candidates.items() if k in bindings.params})
    params = {k: v for k, v in params.items() if k in bindings.params}
    runner = JobRunner(
        cfg, app_ctx.executor, app_ctx.comfy, StateStore(config_dir() / "bench-state"), progress=app_ctx.out.progress
    )
    timings = []
    for i in range(repeat):
        out_dir = remote_join(cfg.data_root, "_bench", engine.id, f"run{i}")
        params["seed"] = i + 1
        record = runner.run(JobSpec(engine, dict(params), out_dir, f"bench {engine.id} #{i + 1}"), force=True)
        timings.append(record["duration_s"])
    avg = sum(timings) / len(timings)
    entry = {
        "engine": engine.id,
        "profile": profile,
        "width": width,
        "height": height,
        "frames": length,
        "seconds": timings,
        "seconds_per_window": round(avg, 1),
        "seconds_per_frame": round(avg / length, 2),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = config_dir() / "bench.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[f"{engine.id}@{profile}@{height}p"] = entry
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    app_ctx.out.result(entry, f"{engine.id} {profile} {width}x{height}x{length}: {avg:.1f}s per window")
