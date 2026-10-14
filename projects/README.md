# projects/

Each project lives in `projects/<name>/` on the laptop:

- `project.yaml`: your intent (look, characters, voices, per-shot modes). Edit freely.
- `state/`: machine-written records (probe results, jobs, takes, picks). Don't edit by hand.
- `previews/`: low-resolution contact sheets and side-by-sides pulled from the workstation.

Heavy media (source video, frames, renders, stems) lives on the workstation under
`<data_root>/projects/<name>/`. This folder is gitignored; version it separately if you want history.
