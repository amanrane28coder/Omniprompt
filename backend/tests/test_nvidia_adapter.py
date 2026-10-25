"""NVIDIA adapter tests. No real key, no real HTTP call is made."""
import asyncio

import pytest

from app.adapters import nvidia_adapter
from app.adapters.nvidia_adapter import NvidiaAdapter
from app.services import orchestration
from app.services.prompt_engine import run_engine


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "Hello from NVIDIA"},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.posted = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.posted = (url, headers, json)
        return _FakeResponse()


def test_nvidia_adapter_normalizes_openai_compatible_response(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "nvapi-test")
    fake = _FakeClient()
    monkeypatch.setattr(
        nvidia_adapter.httpx, "AsyncClient", lambda **kwargs: fake
    )
    adapter = NvidiaAdapter("meta/llama-3.3-70b-instruct")
    result = asyncio.run(adapter.send("Hello", {"temperature": 0.2, "top_p": 1.0}))
    assert result.output_text == "Hello from NVIDIA"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.finish_reason == "stop"
    url, headers, payload = fake.posted
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer nvapi-test"
    assert payload["model"] == "meta/llama-3.3-70b-instruct"


def test_nvidia_adapter_requires_key(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        asyncio.run(NvidiaAdapter("meta/llama-3.3-70b-instruct").send("Hello", {}))


def test_nvidia_models_registered_and_free():
    for model in orchestration.NVIDIA_MODELS:
        assert model in orchestration._ADAPTERS
        assert isinstance(orchestration._ADAPTERS[model], NvidiaAdapter)
        assert orchestration.cost_for(model, 1000, 1000) == 0.0


def test_engine_run_with_nvidia_model_and_no_key_records_error_not_crash(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "")
    # The orchestration module holds adapter instances created at import;
    # its NvidiaAdapter instances read the key at call time from the
    # adapter module, so patching there is enough for the missing-key path
    # only if the adapter reads the module global at call time. It does:
    # NVIDIA_API_KEY is looked up in nvidia_adapter's namespace.
    out = asyncio.run(run_engine(
        "Explain cloud computing.", models=["meta/llama-3.3-70b-instruct"],
        test_input="cloud computing",
    ))
    resp = out["responses"][0]
    assert resp["model"] == "meta/llama-3.3-70b-instruct"
    assert resp["error"] is not None
    assert "NVIDIA_API_KEY" in resp["error"]
    assert resp["scores"] is None
