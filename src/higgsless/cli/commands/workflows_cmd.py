"""`higgsless workflows ...`: validate and render ComfyUI workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from higgsless.cli.context import get_ctx
from higgsless.cli.output import table
from higgsless.engines.bindings import bind
from higgsless.engines.registry import registry
from higgsless.engines.runner import load_workflow
from higgsless.engines.validate import validate_workflow
from higgsless.errors import ValidationError
from higgsless.remote.comfy import ComfyClient

app = typer.Typer(help="Validate and inspect ComfyUI workflows.", no_args_is_help=True)

_PLACEHOLDER = {
    "input_path": "placeholder.png",
    "abs_path": "/placeholder",
    "int": 1,
    "float": 1.0,
    "bool": False,
    "seed": 1,
    "str": "placeholder",
    "json": {},
}


def _object_info(app_ctx: Any, comfy_url: str | None, object_info: Path | None, save: Path | None) -> dict:
    if object_info is not None:
        return json.loads(object_info.read_text(encoding="utf-8"))
    client = ComfyClient(comfy_url) if comfy_url else app_ctx.comfy
    info = client.object_info()
    if save is not None:
        save.write_text(json.dumps(info), encoding="utf-8")
    return info


_NAMED = {"width": 832, "height": 480, "length": 81, "multiplier": 2}


def _dummy_params(engine: Any, bindings: Any, media: dict[str, str] | None = None) -> dict[str, Any]:
    params = engine.params_for("standard")
    params = {k: v for k, v in params.items() if k in bindings.params}
    for name, b in bindings.params.items():
        if name not in params and b.required and b.default is None:
            if b.type == "input_path" and media:
                lowered = name.lower()
                kind = "video" if "video" in lowered else "audio" if "audio" in lowered else "image"
                params[name] = media[kind]
            else:
                params[name] = _NAMED.get(name, _PLACEHOLDER[b.type])
    return params


def _placeholder_media(client: ComfyClient) -> dict[str, str]:
    """Upload a tiny image, video and audio file so loaders validate on the server."""
    import io
    import math
    import struct
    import wave

    from PIL import Image

    files: dict[str, tuple[str, bytes, str]] = {}
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 130, 140)).save(buf, format="PNG")
    files["image"] = ("hl_placeholder.png", buf.getvalue(), "image/png")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 8))) for i in range(16000)))
    files["audio"] = ("hl_placeholder.wav", buf.getvalue(), "audio/wav")
    video = _tiny_video()
    files["video"] = ("hl_placeholder.mp4", video, "video/mp4")
    names = {}
    for kind, (name, data, mime) in files.items():
        resp = client._request(
            "POST", "/upload/image", files={"image": (name, data, mime)}, data={"overwrite": "true", "type": "input"}
        )
        resp.raise_for_status()
        names[kind] = resp.json()["name"]
    return names


def _tiny_video() -> bytes:
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if not shutil.which("ffmpeg"):
        raise ValidationError("ffmpeg is needed on this machine for --server checks")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "v.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=64x64:r=24",
                "-frames:v",
                "9",
                "-pix_fmt",
                "yuv420p",
                str(dst),
            ],
            check=True,
        )
        return dst.read_bytes()


def _server_check(client: ComfyClient, workflow: dict[str, Any]) -> list[dict[str, str]]:
    """Let the server validate a workflow; ignore missing model files on validator machines."""
    resp = client._request("POST", "/prompt", json={"prompt": workflow, "client_id": client.client_id})
    if resp.status_code == 200:
        prompt_id = resp.json().get("prompt_id")
        client._request("POST", "/queue", json={"delete": [prompt_id]})
        client._request("POST", "/interrupt")
        return []
    body = resp.json()
    issues = []
    for node_id, entry in (body.get("node_errors") or {}).items():
        for err in entry.get("errors", []):
            if err.get("type") == "value_not_in_list" and str(err.get("details", "")).split(":")[0].strip() in {
                "unet_name",
                "clip_name",
                "vae_name",
                "lora_name",
                "ckpt_name",
                "model_name",
                "name",
                "audio_encoder_name",
                "vitpose_model",
                "yolo_model",
            }:
                continue
            issues.append(
                {
                    "level": "error",
                    "node": f"{node_id} ({entry.get('class_type')})",
                    "message": f"server: {err.get('message')}: {err.get('details')}",
                }
            )
    if not issues and not body.get("node_errors"):
        issues.append({"level": "error", "node": "-", "message": f"server: {body.get('error')}"})
    return issues


@app.command("validate")
def validate(
    ctx: typer.Context,
    engine_ids: list[str] = typer.Option([], "--engine", "-e", help="Only these engines."),
    comfy_url: str | None = typer.Option(None, help="ComfyUI URL (default from config)."),
    object_info: Path | None = typer.Option(None, help="Validate against a saved /object_info JSON."),
    save_object_info: Path | None = typer.Option(None, help="Save the fetched /object_info here."),
    strict: bool = typer.Option(False, help="Treat unknown combo values (e.g. missing model files) as errors."),
    server: bool = typer.Option(False, help="Also let the server validate each bound workflow via /prompt."),
) -> None:
    """Check every ComfyUI engine's workflow against the server's node definitions."""
    app_ctx = get_ctx(ctx)
    info = _object_info(app_ctx, comfy_url, object_info, save_object_info)
    client = (ComfyClient(comfy_url) if comfy_url else app_ctx.comfy) if server else None
    media = _placeholder_media(client) if client else None
    engines = [e for e in registry().engines.values() if e.backend == "comfyui"]
    if engine_ids:
        engines = [registry().get(i) for i in engine_ids]
    report: dict[str, Any] = {}
    errors = 0
    for engine in sorted(engines, key=lambda e: e.id):
        workflow, bindings = load_workflow(engine)
        issues = validate_workflow(workflow, info, strict_combos=strict)
        bound = bind(workflow, bindings, _dummy_params(engine, bindings, media))
        bound_issues = [i for i in validate_workflow(bound.workflow, info, strict_combos=strict) if i.level == "error"]
        seen = {(i.node, i.message) for i in issues}
        issues += [i for i in bound_issues if (i.node, i.message) not in seen]
        entry = {
            "errors": [i.to_dict() for i in issues if i.level == "error"],
            "warnings": [i.to_dict() for i in issues if i.level == "warning"],
            "bypassed_in_minimal_variant": bound.bypassed,
        }
        if client is not None:
            entry["errors"] += _server_check(client, bound.workflow)
        report[engine.id] = entry
        errors += len(entry["errors"])
    rows = [[k, len(v["errors"]), len(v["warnings"])] for k, v in report.items()]
    if errors:
        raise ValidationError(f"{errors} workflow error(s)", details=report)
    app_ctx.out.result(report, table("workflow validation", ["engine", "errors", "warnings"], rows))


@app.command("render")
def render(
    ctx: typer.Context,
    engine_id: str,
    param: list[str] = typer.Option([], "--param", "-P", help="key=value (JSON values allowed)."),
    profile: str = typer.Option("standard", help="Profile whose defaults apply."),
) -> None:
    """Print the bound API workflow for an engine (useful for debugging in the ComfyUI editor)."""
    app_ctx = get_ctx(ctx)
    engine = registry().get(engine_id)
    workflow, bindings = load_workflow(engine)
    overrides: dict[str, Any] = {}
    for item in param:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    params = {k: v for k, v in engine.params_for(profile, overrides).items() if k in bindings.params}
    bound = bind(workflow, bindings, params)
    app_ctx.out.result({"workflow": bound.workflow, "outputs": bound.outputs, "bypassed": bound.bypassed})
