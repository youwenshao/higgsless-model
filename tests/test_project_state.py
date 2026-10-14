import pytest

from higgsless.core.cache import hash_obj
from higgsless.core.licence import Licence, check_licence
from higgsless.core.manifest import Project, Usage
from higgsless.core.project import create_project, open_project, remote_join
from higgsless.core.state import StateStore
from higgsless.errors import ConfigError, ValidationError


def test_create_edit_preserves_comments(projects_home):
    handle = create_project("demo", usage="public_noncommercial", look="new-world", profile="draft")
    assert open_project(None).root == handle.root
    with handle.edit() as data:
        data["characters"]["ava"] = {"prompt": "a silver-haired pop singer"}
        data["cast"]["P1"] = "ava"
    text = handle.manifest_path.read_text(encoding="utf-8")
    assert "# original (keep sets, swap performers)" in text
    project = handle.load(refresh=True)
    assert project.cast == {"P1": "ava"}
    assert project.resolution == "480p"


def test_invalid_edit_is_rejected_and_file_unchanged(projects_home):
    handle = create_project("demo2", usage="private", look="original", profile="standard")
    before = handle.manifest_path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError) as err, handle.edit() as data:
        data["cast"]["P1"] = "nobody"
    assert "unknown character" in str(err.value.details)
    assert handle.manifest_path.read_text(encoding="utf-8") == before


def test_shot_and_voice_validation():
    base = {"project": "x"}
    with pytest.raises(ValueError):
        Project.model_validate(
            {**base, "shots": [{"id": "s001", "frames": [0, 10]}, {"id": "s002", "frames": [5, 20]}]}
        )
    with pytest.raises(ValueError):
        Project.model_validate({**base, "voices": {"v": {"kind": "designed", "consent": "self", "describe": "x"}}})
    ok = Project.model_validate(
        {**base, "voices": {"v": {"kind": "designed", "consent": "synthetic", "describe": "x"}}}
    )
    assert ok.voices["v"].kind.value == "designed"
    with pytest.raises(ValueError):
        Project.model_validate({**base, "shots": [{"id": "shot1", "frames": [0, 10]}]})


def test_state_store_roundtrip(tmp_path):
    store = StateStore(tmp_path / "state")
    store.write("jobs/abc", {"status": "done"})
    assert store.read("jobs/abc") == {"status": "done"}
    assert store.keys("jobs") == ["jobs/abc"]
    with pytest.raises(ValueError):
        store.read("../escape")


def test_hash_is_order_independent():
    assert hash_obj({"a": 1, "b": [1, 2]}) == hash_obj({"b": [1, 2], "a": 1})


def test_licence_gate():
    nc = Licence(weights="CC-BY-NC-4.0", commercial=False)
    project = Project.model_validate({"project": "x", "usage": "public_noncommercial"})
    with pytest.raises(ValidationError):
        check_licence("svc.yingmusic", nc, project)
    allowed = project.model_copy(update={"allow_noncommercial": ["svc.yingmusic"]})
    assert check_licence("svc.yingmusic", nc, allowed)
    commercial = project.model_copy(update={"usage": Usage.COMMERCIAL, "allow_noncommercial": ["svc.yingmusic"]})
    with pytest.raises(ValidationError):
        check_licence("svc.yingmusic", nc, commercial)


def test_remote_join_flavours():
    assert remote_join("/srv/higgsless", "projects", "p") == "/srv/higgsless/projects/p"
    assert remote_join("C:\\data", "projects", "p") == "C:\\data\\projects\\p"
