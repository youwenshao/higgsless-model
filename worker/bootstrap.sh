#!/usr/bin/env bash
# One-shot, re-runnable setup for the higgsless GPU workstation (Ubuntu 22.04/24.04, RTX 50-series).
#
#   sudo -v && bash worker/bootstrap.sh            # everything
#   STEPS="comfyui nodes service" bash worker/bootstrap.sh
#
# Steps: preflight packages uv dirs comfyui sage nodes service docker tailscale firewall swap summary
# Settings (environment variables):
#   DATA_ROOT=/srv/higgsless  COMFY_ROOT=/opt/comfyui  COMFY_PORT=8188  BIND_ADDR=<tailscale ip>
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130  PYTHON_VERSION=3.12  SWAP_GB=32
#   TS_AUTHKEY=<optional tailscale auth key>  COMFY_EXTRA_ARGS="--fast"
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/srv/higgsless}
COMFY_ROOT=${COMFY_ROOT:-/opt/comfyui}
COMFY_PORT=${COMFY_PORT:-8188}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
SWAP_GB=${SWAP_GB:-32}
COMFY_EXTRA_ARGS=${COMFY_EXTRA_ARGS:-}
STEPS=${STEPS:-"preflight packages uv dirs comfyui sage nodes service docker tailscale firewall swap summary"}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER=${SUDO_USER:-$USER}
UV="$HOME/.local/bin/uv"
PY="$COMFY_ROOT/.venv/bin/python"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

step_preflight() {
  log "preflight"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "needs Linux x86_64"
  [[ $EUID -ne 0 ]] || die "run as your normal user (sudo is used where needed)"
  sudo -v || die "needs sudo"
  has nvidia-smi || die "NVIDIA driver missing: install driver r580 or newer (sudo ubuntu-drivers install), reboot, rerun"
  local driver major
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  major=${driver%%.*}
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $driver"
  (( major >= 580 )) || die "driver $driver is too old for CUDA 13 (cu130 PyTorch): install r580+"
  local ram_gb
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  echo "RAM: ${ram_gb} GB"
  (( ram_gb >= 60 )) || warn "under 64 GB RAM: use INT8/NVFP4 weights and keep SWAP_GB >= 32"
}

step_packages() {
  log "system packages"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends git curl ca-certificates build-essential ffmpeg jq ufw \
    python3-dev libgl1 libglib2.0-0 pkg-config
}

step_uv() {
  log "uv"
  if [[ ! -x "$UV" ]]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
  "$UV" --version
}

step_dirs() {
  log "data root $DATA_ROOT"
  sudo mkdir -p "$DATA_ROOT"/{projects,_hf,_worker,_tmp,_bench} "$COMFY_ROOT"
  sudo chown -R "$RUN_USER:$RUN_USER" "$DATA_ROOT" "$COMFY_ROOT"
}

torch_check() {
  "$PY" - <<'EOF'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available"
major, minor = torch.cuda.get_device_capability(0)
print(f"device {torch.cuda.get_device_name(0)} sm_{major}{minor}")
EOF
}

step_comfyui() {
  log "ComfyUI in $COMFY_ROOT (PyTorch from $TORCH_INDEX)"
  if [[ ! -d "$COMFY_ROOT/.git" ]]; then git clone https://github.com/Comfy-Org/ComfyUI "$COMFY_ROOT"; else git -C "$COMFY_ROOT" pull --ff-only || warn "ComfyUI pull failed"; fi
  [[ -x "$PY" ]] || "$UV" venv --python "$PYTHON_VERSION" "$COMFY_ROOT/.venv"
  "$UV" pip install --python "$PY" --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX"
  "$UV" pip install --python "$PY" -r "$COMFY_ROOT/requirements.txt"
  "$UV" pip install --python "$PY" comfy-kitchen huggingface_hub hf_transfer || warn "comfy-kitchen install failed (NVFP4 fast path unavailable)"
  torch_check
}

step_sage() {
  log "SageAttention 2.2 (built from source for sm_120)"
  if "$PY" -c "import sageattention, sys; sys.exit(0 if sageattention.__version__.startswith('2') else 1)" 2>/dev/null; then
    echo "already installed"; return
  fi
  if ! has nvcc && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    warn "nvcc not found: installing the CUDA 13.0 toolkit from NVIDIA's apt repo"
    local codename; codename=$(. /etc/os-release && echo "${VERSION_ID//./}")
    curl -fsSLo /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${codename}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -y && sudo apt-get install -y cuda-toolkit-13-0 \
      || { warn "CUDA toolkit install failed; skipping SageAttention (PyTorch SDPA still works)"; return; }
  fi
  export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS="${MAX_JOBS:-8}"
  rm -rf /tmp/SageAttention && git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/SageAttention
  "$UV" pip install --python "$PY" setuptools wheel ninja
  (cd /tmp/SageAttention && "$UV" pip install --python "$PY" --no-build-isolation .) \
    || warn "SageAttention build failed; ComfyUI will use PyTorch SDPA"
}

step_nodes() {
  log "custom node packs ($HERE/custom_nodes.txt)"
  local torch_before; torch_before=$("$PY" -c "import torch; print(torch.__version__)")
  while read -r name url ref; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    local dir="$COMFY_ROOT/custom_nodes/$name"
    if [[ ! -d "$dir/.git" ]]; then git clone "$url" "$dir"; fi
    if [[ -n "${ref:-}" ]]; then git -C "$dir" fetch --depth 1 origin "$ref" && git -C "$dir" checkout -q FETCH_HEAD; else git -C "$dir" pull --ff-only || true; fi
    if [[ -f "$dir/requirements.txt" ]]; then
      # Never let a node pack replace the torch build.
      grep -viE '^\s*(torch|torchvision|torchaudio|xformers)\b' "$dir/requirements.txt" > "/tmp/hl-req-$name.txt" || true
      "$UV" pip install --python "$PY" -r "/tmp/hl-req-$name.txt" || warn "requirements for $name failed"
    fi
  done < "$HERE/custom_nodes.txt"
  local torch_after; torch_after=$("$PY" -c "import torch; print(torch.__version__)")
  [[ "$torch_before" == "$torch_after" ]] || die "a node pack changed torch ($torch_before -> $torch_after); reinstall with: STEPS=comfyui bash $0"
}

bind_addr() {
  if [[ -n "${BIND_ADDR:-}" ]]; then echo "$BIND_ADDR"; return; fi
  if has tailscale && tailscale ip -4 >/dev/null 2>&1; then tailscale ip -4 | head -1; else echo 127.0.0.1; fi
}

step_service() {
  log "systemd unit comfyui.service"
  local addr; addr=$(bind_addr)
  [[ "$addr" == 127.0.0.1 ]] && warn "binding ComfyUI to 127.0.0.1 (set up Tailscale, then rerun STEPS=service)"
  sed -e "s|@USER@|$RUN_USER|g" -e "s|@COMFY_ROOT@|$COMFY_ROOT|g" -e "s|@DATA_ROOT@|$DATA_ROOT|g" \
      -e "s|@BIND_ADDR@|$addr|g" -e "s|@PORT@|$COMFY_PORT|g" -e "s|@EXTRA_ARGS@|$COMFY_EXTRA_ARGS|g" \
      "$HERE/comfyui.service" | sudo tee /etc/systemd/system/comfyui.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now comfyui.service
  sudo systemctl restart comfyui.service
  for _ in $(seq 1 60); do
    if curl -fsS "http://$addr:$COMFY_PORT/system_stats" >/dev/null 2>&1; then echo "ComfyUI is up on http://$addr:$COMFY_PORT"; return; fi
    sleep 2
  done
  warn "ComfyUI did not answer yet; check: journalctl -u comfyui -n 100"
}

step_docker() {
  log "Docker and the NVIDIA Container Toolkit"
  if ! has docker; then curl -fsSL https://get.docker.com | sudo sh; fi
  sudo usermod -aG docker "$RUN_USER"
  if ! has nvidia-ctk; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -y && sudo apt-get install -y nvidia-container-toolkit
  fi
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  sudo docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L || warn "containers cannot see the GPU yet"
}

step_tailscale() {
  log "Tailscale"
  has tailscale || curl -fsSL https://tailscale.com/install.sh | sh
  if ! tailscale ip -4 >/dev/null 2>&1; then
    if [[ -n "${TS_AUTHKEY:-}" ]]; then sudo tailscale up --authkey "$TS_AUTHKEY" --ssh=false; else sudo tailscale up; fi
  fi
  echo "Tailscale address: $(tailscale ip -4 | head -1)  name: $(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')"
}

step_firewall() {
  log "firewall: SSH open, ComfyUI only on tailscale0"
  sudo ufw allow OpenSSH
  sudo ufw allow in on tailscale0 to any port "$COMFY_PORT" proto tcp
  sudo ufw deny "$COMFY_PORT"/tcp
  sudo ufw --force enable
  sudo ufw status verbose | sed -n '1,20p'
}

step_swap() {
  log "swap (${SWAP_GB} GB)"
  (( SWAP_GB > 0 )) || return 0
  local have; have=$(awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if (( have >= SWAP_GB )); then echo "swap already ${have} GB"; return; fi
  sudo fallocate -l "${SWAP_GB}G" /swapfile.higgsless
  sudo chmod 600 /swapfile.higgsless && sudo mkswap /swapfile.higgsless && sudo swapon /swapfile.higgsless
  grep -q swapfile.higgsless /etc/fstab || echo '/swapfile.higgsless none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
}

step_summary() {
  log "done; put this in config/workstation.yaml on the laptop (or run higgsless config init)"
  local host addr
  host=$( (has tailscale && tailscale status --json 2>/dev/null | jq -r '.Self.DNSName' | sed 's/\.$//') || hostname)
  addr=$(bind_addr)
  cat <<EOF
executor: ssh
ssh: {host: $host, user: $RUN_USER}
comfy: {url: http://$addr:$COMFY_PORT}
data_root: $DATA_ROOT
comfy_root: $COMFY_ROOT
comfy_python: $PY
hf_cache: $DATA_ROOT/_hf
worker_dir: $DATA_ROOT/_worker
docker: {user: "$(id -u "$RUN_USER"):$(id -g "$RUN_USER")"}
EOF
  echo
  echo "Next, from the laptop: higgsless --json worker build && higgsless --json models sync && higgsless --json doctor"
}

for step in $STEPS; do "step_$step"; done
