import pytest

from higgsless.core.manifest import Project, Shot
from higgsless.core.models import ModelsLock, models_for
from higgsless.engines.bindings import bind
from higgsless.engines.registry import Registry
from higgsless.engines.router import decide
from higgsless.engines.runner import load_workflow

PLACEHOLDER = {
    "input_path": "x.png",
    "abs_path": "/x",
    "int": 1,
    "float": 1.0,
    "bool": False,
    "seed": 1,
    "str": "x",
    "json": {},
}


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def test_every_engine_uses_locked_models(registry):
    lock = ModelsLock.load()
    for engine in registry.engines.values():
        for profile in ("draft", "standard", "max"):
            for model_id in models_for(engine.models, profile):
                assert model_id in lock.models, f"{engine.id} needs {model_id}"


def test_one_default_per_capability(registry):
    capabilities = {e.capability for e in registry.engines.values()}
    for cap in capabilities:
        defaults = [e.id for e in registry.by_capability(cap) if e.default]
        assert len(defaults) == 1, (cap, defaults)


@pytest.mark.parametrize("profile", ["draft", "standard", "max"])
def test_comfy_engines_bind_minimal_and_full(registry, profile):
    for engine in registry.engines.values():
        if engine.backend != "comfyui":
            continue
        workflow, bindings = load_workflow(engine)
        params = {k: v for k, v in engine.params_for(profile).items() if k in bindings.params}
        for name, b in bindings.params.items():
            if b.required and name not in params:
                params[name] = PLACEHOLDER[b.type]
        minimal = bind(workflow, bindings, params)
        assert minimal.outputs, engine.id
        # Every optional node supplied: nothing bypassed except what the profile turns off.
        full = dict(params)
        for name, b in bindings.params.items():
            if b.optional_nodes and full.get(name) is None and name not in engine.params_for(profile):
                full[name] = PLACEHOLDER[b.type]
        bind(workflow, bindings, full)


def test_models_referenced_by_workflows_are_locked(registry):
    lock = ModelsLock.load()
    names = {m.filename for m in lock.models.values()}
    loader_inputs = {
        "unet_name",
        "clip_name",
        "vae_name",
        "lora_name",
        "ckpt_name",
        "model_name",
        "name",
        "audio_encoder_name",
        "vitpose_model",
        "yolo_model",
    }
    for engine in registry.engines.values():
        if engine.backend != "comfyui":
            continue
        workflow, _ = load_workflow(engine)
        for node in workflow.values():
            for key, value in node["inputs"].items():
                if key in loader_inputs and isinstance(value, str):
                    assert value in names, f"{engine.id}: {node['class_type']}.{key}={value} is not in the lockfile"


def _project(**kw) -> Project:
    base = {"project": "x", "characters": {"ava": {"prompt": "a"}, "bo": {"prompt": "b", "stylised": True}}}
    base.update(kw)
    return Project.model_validate(base)


def test_router_modes_engines_and_lipsync(registry):
    shot = {"id": "s001", "frames": [0, 50]}
    one = {"people": [{"track": "1", "presence": 1.0}], "faces": {"max_frac": 0.3}}
    two = {"people": [{"track": "1", "presence": 1.0}, {"track": "2", "presence": 0.9}], "faces": {"max_frac": 0.05}}
    project = _project(look="original", cast={"P1": "ava", "P2": "bo"})
    d = decide(project, Shot.model_validate(shot), one, registry, {"1": "P1"})
    assert (d.mode, d.engine, d.lipsync, d.lipsync_engine) == (
        "swap",
        "swap.wan22_animate_mix",
        True,
        "lipsync.latentsync16",
    )
    d = decide(project, Shot.model_validate(shot), two, registry, {"1": "P1", "2": "P2"})
    assert (d.mode, d.engine, d.lipsync) == ("swap", "swap.scail2", False)

    rebuild = _project(look="new-world", cast={"P1": "bo"})
    d = decide(rebuild, Shot.model_validate(shot), one, registry, {"1": "P1"})
    assert (d.mode, d.engine, d.lipsync_engine) == ("rebuild", "rebuild.wan_animate2", "lipsync.infinitetalk")

    d = decide(project, Shot.model_validate(shot), {"people": [], "faces": {}}, registry, {})
    assert d.mode == "passthrough" and d.engine is None
    d = decide(project, Shot.model_validate({**shot, "mode": "restyle"}), one, registry, {"1": "P1"})
    assert d.engine == "restyle.bernini_v2v"
