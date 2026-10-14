from higgsless.core.config import WorkstationConfig
from higgsless.core.models import ModelsLock, models_for, remote_path, sync_script


def test_lock_parses_and_has_hashes():
    lock = ModelsLock.load()
    assert len(lock.models) > 30
    for model_id, model in lock.models.items():
        if model.dest != "hf":
            assert model.sha256 and len(model.sha256) == 64, model_id
            assert model.size and model.size > 0, model_id


def test_profile_pinned_refs():
    assert models_for(["a", "b@max", "c@draft"], "max") == ["a", "b"]
    assert models_for(["a", "b@max", "c@draft"], "draft") == ["a", "c"]


def test_sync_script_is_resumable_curl():
    cfg = WorkstationConfig(executor="local")
    model = ModelsLock.load().get("wan_animate2_int8")
    assert remote_path(cfg, model) == "/opt/comfyui/models/diffusion_models/wan_animate_2_int8_convrot.safetensors"
    script = sync_script(cfg, model, verify=True)
    assert "curl -L --fail" in script and "-C -" in script and "sha256sum -c" in script
