"""`higgsless config ...`: workstation connection settings."""

from __future__ import annotations

import io

import typer
from ruamel.yaml import YAML

from higgsless.cli.context import get_ctx
from higgsless.core.config import load_config
from higgsless.core.paths import config_dir
from higgsless.errors import ConfigError

app = typer.Typer(help="Workstation connection config.", no_args_is_help=True)


@app.command("show")
def show(ctx: typer.Context) -> None:
    """Print the effective workstation config."""
    app_ctx = get_ctx(ctx)
    data = app_ctx.config.model_dump(mode="json")
    app_ctx.out.result(data)


@app.command("init")
def init(
    ctx: typer.Context,
    host: str = typer.Option(..., help="Workstation hostname, Tailscale name or SSH alias."),
    comfy_url: str | None = typer.Option(None, help="ComfyUI URL (default http://<host>:8188)."),
    user: str | None = typer.Option(None, help="SSH user."),
    key_path: str | None = typer.Option(None, help="SSH private key."),
    data_root: str = typer.Option("/srv/higgsless", help="Data root on the workstation."),
    force: bool = typer.Option(False, help="Overwrite an existing config."),
) -> None:
    """Write config/workstation.yaml from the example."""
    app_ctx = get_ctx(ctx)
    target = config_dir() / "workstation.yaml"
    if target.exists() and not force:
        raise ConfigError(f"{target} exists", hint="pass --force to overwrite")
    yaml = YAML()
    data = yaml.load((config_dir() / "workstation.example.yaml").read_text(encoding="utf-8"))
    data["ssh"]["host"] = host
    data["ssh"]["user"] = user
    data["ssh"]["key_path"] = key_path
    data["comfy"]["url"] = comfy_url or f"http://{host}:8188"
    data["data_root"] = data_root
    data["hf_cache"] = f"{data_root.rstrip('/')}/_hf"
    data["worker_dir"] = f"{data_root.rstrip('/')}/_worker"
    buf = io.StringIO()
    yaml.dump(data, buf)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    cfg = load_config(target)
    app_ctx.out.result({"path": str(target), "config": cfg.model_dump(mode="json")}, f"wrote {target}")
