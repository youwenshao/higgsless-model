"""`higgsless worker ...`: deploy worker code and build tool images on the workstation."""

from __future__ import annotations

import typer

from higgsless.cli.context import get_ctx
from higgsless.core.project import remote_join
from higgsless.remote.deploy import deploy_worker

app = typer.Typer(help="Deploy worker code and build Docker tool images.", no_args_is_help=True)

IMAGES = ("tools", "soulx", "applio", "voxcpm", "acestep")


@app.command("deploy")
def deploy(ctx: typer.Context, force: bool = typer.Option(False, help="Copy even if unchanged.")) -> None:
    """Copy src/higgsless, worker/ and engines/ to the workstation's worker_dir."""
    app_ctx = get_ctx(ctx)
    result = deploy_worker(app_ctx.executor, app_ctx.config.worker_dir, force=force)
    app_ctx.out.result(result, "deployed" if result["deployed"] else "worker code already up to date")


@app.command("build")
def build(
    ctx: typer.Context,
    images: list[str] = typer.Option(list(IMAGES), "--image", "-i", help="Images to build."),
    no_cache: bool = typer.Option(False, help="Pass --no-cache to docker build."),
) -> None:
    """Build the Docker tool images on the workstation (deploys worker code first)."""
    app_ctx = get_ctx(ctx)
    cfg = app_ctx.config
    deploy_worker(app_ctx.executor, cfg.worker_dir)
    built = []
    for name in images:
        tag = cfg.docker.images[name]
        dockerfile = remote_join(cfg.worker_dir, "worker", "docker", name, "Dockerfile")
        argv = ["docker", "build", "-t", tag, "-f", dockerfile]
        if no_cache:
            argv.append("--no-cache")
        argv.append(cfg.worker_dir)
        app_ctx.out.progress(f"building {tag}")
        res = app_ctx.executor.run(argv, timeout=4 * 3600, on_line=app_ctx.out.progress if app_ctx.verbose else None)
        built.append({"image": tag, "seconds": round(res.duration_s, 1)})
    app_ctx.out.result({"built": built}, f"built {len(built)} image(s)")


@app.command("status")
def status(ctx: typer.Context) -> None:
    """Show which tool images exist on the workstation and whether containers see the GPU."""
    app_ctx = get_ctx(ctx)
    cfg = app_ctx.config
    images = {}
    for name, tag in cfg.docker.images.items():
        res = app_ctx.executor.run(["docker", "image", "inspect", "--format", "{{.Id}}", tag], check=False)
        images[name] = {"tag": tag, "present": res.ok}
    gpu = None
    if images.get("tools", {}).get("present"):
        res = app_ctx.executor.run(
            ["docker", "run", "--rm", "--gpus", cfg.docker.gpus, cfg.docker.images["tools"], "nvidia-smi", "-L"],
            check=False,
            timeout=120,
        )
        gpu = res.stdout.strip() if res.ok else f"error: {res.stderr.strip()[:300]}"
    app_ctx.out.result({"images": images, "container_gpu": gpu})
