# AGENTS.md

How coding agents should drive `higgsless`. Humans can follow the same steps.

## Ground rules

- Always pass `--json` and read the single JSON document on stdout. Progress is on stderr.
- Exit codes: 0 ok, 3 config, 4 remote unreachable, 5 validation, 6 job failed, 7 review gate needs a
  decision. On 7, inspect the previews named in the error details, then approve or pick.
- Edit intent in `projects/<name>/project.yaml` (or with `shots set`, `cast assign`). Never edit
  `projects/<name>/state/` by hand.
- Commands are idempotent and cached; re-running resumes. Use `--force` only to redo finished work.
- Never clone a real person's voice without consent recorded in the voice profile
  (`consent: self | written`). Prefer `kind: designed` voices for public uploads.

## Setup check

```
higgsless --json doctor          # SSH, ComfyUI, GPU, node packs, models, containers
higgsless --json models sync --profile standard
higgsless --json bench           # measures seconds per window on this workstation
```

## Pipeline

```
higgsless --json init my-mv --source path/to/mv.mp4 --usage public_noncommercial --look new-world
higgsless --json shots detect
higgsless --json analyze                 # person tracks, face sizes, captions
higgsless --json cast detect
higgsless --json cast assign P1 ava
higgsless --json char create ava --prompt "..."
higgsless --json keyframe make --all     # review previews/keyframes/*.jpg, then:
higgsless --json keyframe approve --all
higgsless --json shots suggest           # explains the auto router's choice per shot
higgsless --json render --all --takes 2
higgsless --json review --all            # contact sheets and metrics per take
higgsless --json pick s012 --take 2
higgsless --json audio separate
higgsless --json audio transcribe
higgsless --json lyrics fit              # edit audio/lyrics_new.yaml until it reports no mismatches
higgsless --json voice create lead --kind designed --describe "warm alto pop vocalist"
higgsless --json audio sing
higgsless --json audio mix
higgsless --json lipsync --all
higgsless --json finish --all
higgsless --json conform
```

Or run everything that is ready: `higgsless --json run --until conform`, and check
`higgsless --json status`.

## Reviewing images

Contact sheets and side-by-sides land in `projects/<name>/previews/`. Open them with your image
reading tool and judge identity, hands, face, background integrity and lip-sync before picking.
