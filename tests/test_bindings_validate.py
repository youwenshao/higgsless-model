import pytest

from higgsless.engines.bindings import Bindings, bind
from higgsless.engines.validate import validate_workflow
from higgsless.errors import ValidationError

WORKFLOW = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "m.safetensors", "weight_dtype": "default"},
        "_meta": {"title": "Model"},
    },
    "2": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["1", 0], "lora_name": "x", "strength_model": 1.0},
        "_meta": {"title": "Speed LoRA"},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {"model": ["2", 0], "seed": 0, "steps": 4},
        "_meta": {"title": "Sampler"},
    },
    "4": {
        "class_type": "SaveImage",
        "inputs": {"images": ["3", 0], "filename_prefix": "x"},
        "_meta": {"title": "Save Frames"},
    },
}

BINDINGS = Bindings.model_validate(
    {
        "params": {
            "steps": {"target": "Sampler.steps", "type": "int", "required": True},
            "seed": {"target": "Sampler.seed", "type": "seed"},
            "speed_lora": {"target": "Speed LoRA.lora_name", "type": "str", "optional_node": "Speed LoRA"},
        },
        "bypass": {"Speed LoRA": {0: "model"}},
        "outputs": {"frames": {"node": "Save Frames"}},
    }
)


def test_bind_sets_values_and_prefix():
    bound = bind(
        WORKFLOW,
        BINDINGS,
        {"steps": "6", "seed": 7, "speed_lora": "lightx2v.safetensors"},
        {"frames": "projects/p/renders/s001/frames"},
    )
    wf = bound.workflow
    assert wf["3"]["inputs"]["steps"] == 6 and wf["3"]["inputs"]["seed"] == 7
    assert wf["2"]["inputs"]["lora_name"] == "lightx2v.safetensors"
    assert wf["4"]["inputs"]["filename_prefix"] == "projects/p/renders/s001/frames"
    assert bound.outputs == {"frames": "4"}
    assert WORKFLOW["3"]["inputs"]["steps"] == 4  # original untouched


def test_bypass_rewires_passthrough():
    bound = bind(WORKFLOW, BINDINGS, {"steps": 4})
    assert "2" not in bound.workflow
    assert bound.workflow["3"]["inputs"]["model"] == ["1", 0]
    assert bound.bypassed == ["Speed LoRA"]


def test_missing_required_and_unknown_params_fail():
    with pytest.raises(ValidationError):
        bind(WORKFLOW, BINDINGS, {})
    with pytest.raises(ValidationError):
        bind(WORKFLOW, BINDINGS, {"steps": 4, "stpes": 5})


OBJECT_INFO = {
    "UNETLoader": {
        "input": {"required": {"unet_name": [["m.safetensors"]], "weight_dtype": [["default", "fp8"]]}},
        "output": ["MODEL"],
    },
    "LoraLoaderModelOnly": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "lora_name": [["x"]],
                "strength_model": ["FLOAT", {"min": -100, "max": 100}],
            }
        },
        "output": ["MODEL"],
    },
    "KSampler": {
        "input": {"required": {"model": ["MODEL"], "seed": ["INT", {"min": 0}], "steps": ["INT", {"min": 1}]}},
        "output": ["LATENT"],
    },
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}, "output": []},
}


def test_validate_finds_type_mismatch_and_unknowns():
    issues = validate_workflow(WORKFLOW, OBJECT_INFO)
    messages = [i.message for i in issues if i.level == "error"]
    assert any("expects IMAGE, linked to LATENT" in m for m in messages)
    wf = {**WORKFLOW, "9": {"class_type": "Nope", "inputs": {}}}
    assert any("unknown node class" in i.message for i in validate_workflow(wf, OBJECT_INFO))


def test_validate_combo_values_warn_by_default():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing.safetensors", "weight_dtype": "default"}}}
    issues = validate_workflow(wf, OBJECT_INFO)
    assert [i.level for i in issues] == ["warning"]
    assert [i.level for i in validate_workflow(wf, OBJECT_INFO, strict_combos=True)] == ["error"]
