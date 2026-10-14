"""Tasks that run on the workstation (inside the tools container, or natively).

Each task is `python -m higgsless.worker_tasks.<name> --result R --output-dir D --params JSON` and
writes `{"ok": true, "outputs": {...}, ...}` to R. Heavy dependencies are imported inside tasks.
"""
