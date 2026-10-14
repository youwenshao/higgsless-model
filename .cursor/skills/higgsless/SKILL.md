---
name: higgsless
description: Drive the higgsless music-video pipeline. Use when working in higgsless-model, regenerating a music video, or running ComfyUI engines on the workstation.
---

# higgsless

Run every command as `higgsless --json ...` and read the one JSON document on stdout. Progress is on stderr.
Exit codes: 3 config, 4 the workstation is unreachable, 5 validation, 6 a job failed, 7 a review gate.

- Intent lives in `projects/<name>/project.yaml`. Machine state lives in `projects/<name>/state/`; never edit it.
- A 7 means look at the contact sheet named in the error, then `keyframe approve` or `pick`.
- Voices cloned from a real person need `consent: self` or `consent: written`. Prefer `--kind designed`.

Order: `init`, `shots detect`, `analyze`, `cast assign`, `char create`, `keyframe make`, `keyframe approve`,
`render`, `audio separate`, `audio transcribe`, `lyrics fit`, `voice create`, `audio sing`, `audio mix`,
`lipsync`, `finish`, `conform`. `run --until conform` does this and stops at the first gate. `status` says where
a project is. `doctor` checks the workstation.
