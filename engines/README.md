# engines/

One YAML manifest per engine. Each declares its capability, backend (a ComfyUI workflow with
bindings, or a Docker tool command), constraints, required model files, quality profiles and licence.

`higgsless engines list --json` shows them; `higgsless workflows validate` checks the ComfyUI ones
against a live server's node definitions.
