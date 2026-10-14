import json
import subprocess
import sys

from tests.conftest import REPO


def run_cli(*args: str, env: dict | None = None) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "higgsless", "--json", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
        env=env,
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout) if proc.stdout.strip() else {}


def test_version_json():
    code, data = run_cli("version")
    assert code == 0 and data["ok"] and data["data"]["version"]


def test_missing_config_is_exit_code_3(tmp_path):
    import os

    env = {**os.environ, "HIGGSLESS_CONFIG": str(tmp_path / "nope.yaml")}
    code, data = run_cli("config", "show", env=env)
    assert code == 3
    assert data["ok"] is False and data["error"]["kind"] == "config"
