#!/usr/bin/env bash
# CPU-only ComfyUI for validating workflows on a machine without the GPU workstation (Linux or WSL).
#
#   bash scripts/dev/setup_comfy_validator.sh
#   ~/higgsless/comfy-validator/ComfyUI/.venv/bin/python ~/higgsless/comfy-validator/ComfyUI/main.py \
#       --cpu --listen 127.0.0.1 --port 8189 --disable-api-nodes
#   higgsless workflows validate --comfy-url http://127.0.0.1:8189
set -euo pipefail
TARGET=${TARGET:-$HOME/higgsless/comfy-validator}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="$HOME/.local/bin/uv"

mkdir -p "$TARGET"
cd "$TARGET"
[[ -x "$UV" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
[[ -d ComfyUI/.git ]] || git clone --depth 1 https://github.com/Comfy-Org/ComfyUI ComfyUI
cd ComfyUI
[[ -x .venv/bin/python ]] || "$UV" venv --python 3.12 .venv
PY="$PWD/.venv/bin/python"
"$UV" pip install --python "$PY" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$PY" -r requirements.txt

# Required and optional node packs from worker/custom_nodes.txt.
sed -E 's/^# (ComfyUI-)/\1/' "$REPO/worker/custom_nodes.txt" | grep -vE '^\s*(#|$)' | while read -r name url ref; do
  dir="custom_nodes/$name"
  [[ -d "$dir/.git" ]] || git clone --depth 1 "$url" "$dir"
  if [[ -f "$dir/requirements.txt" ]]; then
    grep -viE '^\s*(torch|torchvision|torchaudio|xformers)\b' "$dir/requirements.txt" > "/tmp/hl-req-$name.txt" || true
    "$UV" pip install --python "$PY" -r "/tmp/hl-req-$name.txt" || echo "warning: requirements for $name failed"
  fi
done
echo "ready: $PY $PWD/main.py --cpu --listen 127.0.0.1 --port 8189 --disable-api-nodes"
