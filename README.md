# higgsless-model

> Higgsfield without the Higgs: zero credits, zero bosons, one very warm RTX 5090.

`higgsless` is an agent-driven CLI that regenerates a music video with new characters and new
vocals using open models. It is a local, open-source stand-in for Higgsfield Genjutsu:

- **Motion Transfer** (same motion, new scene) → Wan-Animate-2, SCAIL-2 animation mode
- **Object Swap** (keep the shot, replace the performer) → SCAIL-2 replacement, Wan2.2-Animate Mix, Bernini-R
- **Restyle** → Bernini-R video-to-video, Wan2.2-VACE-Fun
- **Character sheets and sample scenes** → Qwen-Image 2.1, Qwen-Image-Edit-2511
- **New lyrics in a cloned or designed voice** → RoFormer separation, SoulX-Singer, VoxCPM2, Applio
- **Lip-sync** → LatentSync 1.6, InfiniteTalk

The CLI runs on your laptop. All GPU work runs on a Linux workstation (RTX 5090) that hosts
ComfyUI and a few Docker tool containers, reached over Tailscale or the LAN.

`higgsless --json status` shows how far a project has got, and `higgsless --json run --until conform` runs every
stage that is ready, stopping when a keyframe or a lyric line needs a decision.

The commands, in order: `init`, `ingest`, `shots detect`, `analyze`, `cast assign`, `char create`,
`keyframe make`, `keyframe approve`, `render`, `audio separate`, `audio transcribe`, `lyrics fit`,
`voice create`, `audio sing`, `audio mix`, `lipsync`, `finish`, `conform`.

`higgsless --json doctor` checks the workstation. GPU setup is `worker/bootstrap.sh` on the Linux machine.
The design is in [docs/superpowers/specs/2026-10-14-higgsless-model-design.md](docs/superpowers/specs/2026-10-14-higgsless-model-design.md)
and [AGENTS.md](AGENTS.md) shows the same flow for an agent.
