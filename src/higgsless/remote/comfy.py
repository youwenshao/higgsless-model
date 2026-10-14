"""ComfyUI HTTP and WebSocket client."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from higgsless.errors import JobFailed, RemoteError, ValidationError

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class OutputFile:
    node_id: str
    kind: str  # images | gifs | audio | videos | ...
    filename: str
    subfolder: str
    type: str  # output | temp | input

    @property
    def rel_path(self) -> str:
        """Path relative to ComfyUI's output directory."""
        return f"{self.subfolder}/{self.filename}" if self.subfolder else self.filename

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id, "kind": self.kind, "path": self.rel_path, "type": self.type}


class ComfyClient:
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 6 * 3600,
        poll_s: float = 2.0,
        transport: httpx.BaseTransport | None = None,
        client_id: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.client_id = client_id or uuid.uuid4().hex
        self._http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(30.0, read=300.0), transport=transport)
        self._ws_enabled = transport is None

    # HTTP helpers -----------------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise RemoteError(
                f"ComfyUI unreachable at {self.url}: {exc}",
                hint="check comfy.url, Tailscale, and `systemctl status comfyui` on the workstation",
            ) from exc

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise RemoteError(f"{method} {path} returned HTTP {resp.status_code}", details=resp.text[:2000])
        return resp.json() if resp.content else None

    def system_stats(self) -> dict[str, Any]:
        return self._json("GET", "/system_stats")

    def object_info(self) -> dict[str, Any]:
        return self._json("GET", "/object_info")

    def queue_status(self) -> dict[str, Any]:
        return self._json("GET", "/queue")

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self._json("POST", "/free", json={"unload_models": unload_models, "free_memory": free_memory})

    def interrupt(self) -> None:
        self._json("POST", "/interrupt")

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self._json("GET", f"/history/{prompt_id}") or {}
        return data.get(prompt_id)

    # Prompts ----------------------------------------------------------------------------------
    def queue_prompt(self, workflow: dict[str, Any], extra_data: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        if extra_data:
            payload["extra_data"] = extra_data
        resp = self._request("POST", "/prompt", json=payload)
        if resp.status_code == 400:
            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text[:2000]}
            raise ValidationError(
                "ComfyUI rejected the workflow",
                details={"error": body.get("error"), "node_errors": body.get("node_errors")},
                hint="run `higgsless workflows validate` and check that the node packs and models are installed",
            )
        if resp.status_code >= 400:
            raise RemoteError(f"POST /prompt returned HTTP {resp.status_code}", details=resp.text[:2000])
        return resp.json()["prompt_id"]

    def _open_ws(self) -> Any:
        if not self._ws_enabled:
            return None
        try:
            from websockets.sync.client import connect

            ws_url = self.url.replace("https://", "wss://").replace("http://", "ws://")
            return connect(f"{ws_url}/ws?clientId={self.client_id}", open_timeout=10, max_size=None)
        except Exception:  # noqa: BLE001 - fall back to polling
            return None

    def run(
        self, workflow: dict[str, Any], on_progress: ProgressCallback | None = None, timeout_s: float | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Queue a workflow and wait for it. Returns (prompt_id, history entry)."""
        ws = self._open_ws()  # connect first so no progress messages are missed
        try:
            prompt_id = self.queue_prompt(workflow)
            return prompt_id, self._wait(prompt_id, ws, on_progress, timeout_s)
        finally:
            if ws is not None:
                ws.close()

    def wait(self, prompt_id: str, on_progress: ProgressCallback | None = None, timeout_s: float | None = None):
        ws = self._open_ws()
        try:
            return self._wait(prompt_id, ws, on_progress, timeout_s)
        finally:
            if ws is not None:
                ws.close()

    def _wait(self, prompt_id: str, ws: Any, on_progress: ProgressCallback | None, timeout_s: float | None):
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        last_poll = 0.0
        while True:
            if time.monotonic() > deadline:
                raise JobFailed(f"timed out waiting for ComfyUI prompt {prompt_id}")
            finished_hint = False
            if ws is not None:
                try:
                    raw = ws.recv(timeout=self.poll_s)
                except TimeoutError:
                    raw = None
                except Exception:  # noqa: BLE001 - connection dropped; keep polling history
                    ws = None
                    raw = None
                if isinstance(raw, str):
                    finished_hint = self._handle_message(json.loads(raw), prompt_id, on_progress)
            else:
                time.sleep(self.poll_s)
            now = time.monotonic()
            if finished_hint or ws is None or now - last_poll > 15:
                last_poll = now
                entry = self.history(prompt_id)
                if entry is not None:
                    return self._check_entry(prompt_id, entry)

    @staticmethod
    def _handle_message(msg: dict[str, Any], prompt_id: str, on_progress: ProgressCallback | None) -> bool:
        kind = msg.get("type")
        data = msg.get("data") or {}
        if data.get("prompt_id") not in (None, prompt_id):
            return False
        if kind == "progress" and on_progress is not None:
            on_progress({"node": data.get("node"), "value": data.get("value"), "max": data.get("max")})
        elif kind == "executing" and on_progress is not None and data.get("node") is not None:
            on_progress({"node": data.get("node"), "value": None, "max": None})
        elif kind == "execution_error":
            raise JobFailed(
                f"ComfyUI failed in {data.get('node_type')} (node {data.get('node_id')}): "
                f"{str(data.get('exception_message', '')).strip()[:500]}",
                details={"exception_type": data.get("exception_type"), "traceback": data.get("traceback", [])[-8:]},
            )
        elif kind == "execution_interrupted":
            raise JobFailed("ComfyUI execution was interrupted")
        elif kind == "execution_success" or (
            kind == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id
        ):
            return True
        return False

    @staticmethod
    def _check_entry(prompt_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            messages = status.get("messages") or []
            error = next((m[1] for m in messages if m and m[0] == "execution_error"), {})
            raise JobFailed(
                f"ComfyUI prompt {prompt_id} failed in {error.get('node_type')}: "
                f"{str(error.get('exception_message', '')).strip()[:500]}",
                details={"traceback": error.get("traceback", [])[-8:]},
            )
        return entry

    @staticmethod
    def output_files(entry: dict[str, Any]) -> list[OutputFile]:
        files: list[OutputFile] = []
        for node_id, outputs in (entry.get("outputs") or {}).items():
            for kind, items in outputs.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and "filename" in item:
                        files.append(
                            OutputFile(
                                node_id=str(node_id),
                                kind=kind,
                                filename=item["filename"],
                                subfolder=item.get("subfolder", ""),
                                type=item.get("type", "output"),
                            )
                        )
        return files

    def close(self) -> None:
        self._http.close()
