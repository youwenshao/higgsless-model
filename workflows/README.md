# workflows/

ComfyUI workflows in API format (`*.api.json`), one folder per engine, plus a `bindings.yaml`
that maps logical parameters (reference image, driving video, width, seed...) to node inputs by
node title. Every node carries a `_meta.title` so bindings survive re-exports.

To replace a workflow with one exported from the ComfyUI editor ("Export (API)"), keep the node
titles used in `bindings.yaml`, then run `higgsless workflows validate`.
