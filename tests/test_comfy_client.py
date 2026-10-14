import httpx
import pytest

from higgsless.errors import JobFailed, ValidationError
from higgsless.remote.comfy import ComfyClient


def _client(handler) -> ComfyClient:
    return ComfyClient("http://comfy.test", transport=httpx.MockTransport(handler), poll_s=0.01)


def test_run_polls_history_and_collects_outputs():
    calls = {"history": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1", "number": 1, "node_errors": {}})
        if request.url.path == "/history/p1":
            calls["history"] += 1
            if calls["history"] < 2:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "p1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "f_00001_.png", "subfolder": "projects/p/out", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        return httpx.Response(404)

    client = _client(handler)
    prompt_id, entry = client.run({"1": {"class_type": "X", "inputs": {}}})
    assert prompt_id == "p1"
    files = client.output_files(entry)
    assert [f.rel_path for f in files] == ["projects/p/out/f_00001_.png"]


def test_rejected_workflow_raises_validation_error():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}, "node_errors": {"3": {"errors": ["x"]}}})

    with pytest.raises(ValidationError) as err:
        _client(handler).queue_prompt({})
    assert err.value.details["node_errors"] == {"3": {"errors": ["x"]}}


def test_failed_execution_raises_job_failed():
    def handler(request):
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p2"})
        return httpx.Response(
            200,
            json={
                "p2": {
                    "status": {
                        "status_str": "error",
                        "messages": [
                            ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]
                        ],
                    }
                }
            },
        )

    with pytest.raises(JobFailed, match="CUDA out of memory"):
        _client(handler).run({})
