"""Import ComfyUI's comfy_extras modules one by one to find slow or hanging imports.

<comfy>/.venv/Scripts/python.exe scripts/dev/probe_comfy_imports.py <comfy dir>
"""

from __future__ import annotations

import importlib
import os
import sys
import time

comfy = sys.argv[1]
sys.path.insert(0, comfy)
os.chdir(comfy)
sys.argv = [sys.argv[0], "--cpu"]
t0 = time.time()
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import nodes  # noqa: E402,F401

print(f"nodes imported in {time.time() - t0:.1f}s", flush=True)
for name in sorted(os.listdir(os.path.join(comfy, "comfy_extras"))):
    if not name.startswith("nodes_") or not name.endswith(".py"):
        continue
    mod = "comfy_extras." + name[:-3]
    start = time.time()
    print(f"-> {mod}", flush=True)
    try:
        importlib.import_module(mod)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        status = f"ERROR {type(exc).__name__}: {exc}"[:200]
    print(f"   {time.time() - start:.1f}s {status}", flush=True)
print("done", flush=True)
