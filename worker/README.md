# worker/

Everything that runs on the Linux RTX 5090 workstation:

- `bootstrap.sh`: one-shot setup (ComfyUI with PyTorch cu130, SageAttention 2.2, comfy-kitchen,
  custom nodes, Docker with the NVIDIA Container Toolkit, Tailscale, firewall, systemd unit).
- `comfyui.service`: systemd unit template.
- `custom_nodes.txt`: ComfyUI node packs to install.
- `models.lock.yaml`: model files per engine (Hugging Face repo, path, size, sha256, ComfyUI folder).
- `docker/`: tool container images (tools, soulx, applio, voxcpm, acestep).
- `gpl/`: GPL-3.0 scripts (mixing and mastering) that run inside the tools container.
