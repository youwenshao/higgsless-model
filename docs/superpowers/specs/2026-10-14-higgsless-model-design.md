# higgsless-model design

Date: 2026-10-14. Status: approved plan, implementation in progress.

## Goal

Regenerate a ~3-minute music video with different characters and new lyrics sung in a cloned or
designed voice, fully locally, as an open-source stand-in for Higgsfield Genjutsu. The pipeline is
driven primarily by coding agents, so every step is a non-interactive CLI command with JSON output.

## Decisions

- All three Genjutsu modes, selectable per shot, with an `auto` router.
- One or more performers per shot.
- New lyrics (singing voice synthesis plus a lip-sync pass), English first, then Mandarin,
  Cantonese, Japanese and Korean.
- Voice owner undecided, so voices are swappable profiles: reference clip, designed voice, trained RVC.
- Public, non-commercial use. Permissive engines by default; non-commercial engines opt-in through a
  licence gate.
- Code on the Windows laptop; GPU work on a Linux RTX 5090 workstation with 64 GB RAM, reached over
  Tailscale or the LAN.

## Genjutsu to open-source mapping

- Motion Transfer → `rebuild.wan_animate2` (Wan-Animate-2, Apache-2.0), `rebuild.scail2_animate`.
- Object Swap → `swap.scail2` (multi-person, relighting and DPO LoRAs), `swap.wan22_animate_mix`
  (single person, relighting LoRA), `swap.bernini_rv2v` (face, outfit, prop edits).
- Restyle → `restyle.bernini_v2v`, `restyle.vace_fun`.
- Character sheet and sample scene → `sheet.qwen_edit2511` / `sheet.qwen_image21`, `keyframe.qwen_edit2511`.

## Architecture

The laptop runs the `higgsless` CLI (control plane). The workstation runs:

- ComfyUI as a systemd service bound to the Tailscale address. Its `--input-directory` and
  `--output-directory` both point at the data root (`/srv/higgsless`), so workflows read and write
  project files in place and nothing is uploaded through the API.
- Docker containers for tools whose dependencies conflict with ComfyUI's PyTorch build:
  `tools` (ffmpeg, PySceneDetect, TransNetV2, audio-separator, analysis and QA scripts, GPL mixing
  scripts), `soulx` (SoulX-Singer), `applio` (RVC), `voxcpm` (VoxCPM2 voice design), `acestep`
  (ACE-Step 1.5 repaint).

The CLI talks to ComfyUI over HTTP and WebSocket, and to the workstation over SSH (asyncssh exec and
SFTP). Heavy media never leaves the workstation; the laptop pulls previews and deliverables only.

### Units

- `core/`: manifest schema (pydantic), project handle, state store, content-hash cache, rational
  timecode, window planning, licence gate, configuration.
- `remote/`: `Executor` (local or SSH) and `ComfyClient`.
- `engines/`: registry of engine manifests, bindings templating, ComfyUI and Docker adapters,
  workflow validation against `/object_info`, auto router.
- `video/`: probe, ingest, shot detection, extraction, conform, OpenTimelineIO export.
- `analysis/`: person tracks, captions, cast clustering.
- `audio/`: separation, transcription, syllable counting and lyric fitting, voice profiles, singing,
  mixing.
- `qa/`: contact sheets, metrics, review and pick.
- `pipeline/`: stage graph, engine-grouped scheduler, `run` and `status`.
- `worker_tasks/`: entry points executed on the workstation inside the tools container.

### Data model

`projects/<name>/project.yaml` (on the laptop) holds intent: look, profile, characters, voices, cast,
and per-shot `mode`, `engine`, `cast` and `lipsync`. `projects/<name>/state/` holds machine records
(probe, shots, analysis, jobs, takes, picks, stage hashes). The workstation holds media under
`<data_root>/projects/<name>/` (`source/`, `shots/`, `characters/`, `keyframes/`, `renders/`,
`audio/`, `lipsync/`, `finish/`, `conform/`, `previews/`).

### Timing rules

- All timing is integer frames at the source's rational frame rate.
- Sources at 30 fps or less go into generators one frame to one frame and are re-encoded at the
  exact source rate. Sources above 30 fps are decimated by an integer factor and interpolated back.
- Windows are 4n+1 frames. Dimensions snap to multiples of 32.

### Engines

Each engine is `engines/<id>.yaml` with capability, backend, workflow and bindings (ComfyUI) or
image and command (Docker), constraints, models, profiles and licence. Bindings map logical
parameters to node inputs by node title; optional nodes can be bypassed when a parameter is unset.

### Error handling

Errors map to exit codes: 3 config, 4 remote, 5 validation, 6 job failed, 7 review gate. In
`--json` mode every command prints exactly one JSON document on stdout; progress goes to stderr.
Jobs are cached by a hash of engine, workflow, parameters and input fingerprints, so re-running a
command skips finished work and resumes failed work.

### Testing

Unit tests cover timecode maths, window planning, conform frame counts, bindings, the router,
syllable counting and lyric fitting, the licence gate, the ComfyUI client (mock transport) and the
CLI. A CPU-only ComfyUI on the laptop validates workflow node names and inputs. A remote smoke test
runs the full pipeline on a short clip once the workstation is up.

## Risks

- Wan-Animate-2 identity drift over chained windows and unclear camera-move fidelity.
- Multi-person swap quality (SCAIL-2 against multi-pass Wan2.2-Animate).
- SoulX-Singer English quality; lip-sync on singing and stylised faces.
- 64 GB RAM headroom with pose caching.
- No published Wan-Animate-2 timings on an RTX 5090 (`higgsless bench` measures them).
