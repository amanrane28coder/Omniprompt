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
        self.timeout = kwargs.get("timeout")

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
    adapter = NvidiaAdapter("openai/gpt-oss-20b")
    result = asyncio.run(adapter.send("Hello", {"temperature": 0.2, "top_p": 1.0}))
    assert result.output_text == "Hello from NVIDIA"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.finish_reason == "stop"
    url, headers, payload = fake.posted
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer nvapi-test"
    assert payload["model"] == "openai/gpt-oss-20b"


def test_nvidia_adapter_requires_key(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        asyncio.run(NvidiaAdapter("openai/gpt-oss-20b").send("Hello", {}))


def test_nvidia_adapter_uses_120s_timeout(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "nvapi-test")
    fake = _FakeClient()

    def _client(**kwargs):
        fake.timeout = kwargs.get("timeout")
        return fake

    monkeypatch.setattr(nvidia_adapter.httpx, "AsyncClient", _client)
    asyncio.run(NvidiaAdapter("openai/gpt-oss-20b").send("Hello", {}))
    assert fake.timeout == 120


def test_timeout_error_is_named_not_empty(monkeypatch):
    class _TimeoutAdapter:
        async def send(self, prompt_text, params):
            raise asyncio.TimeoutError()

    monkeypatch.setitem(orchestration._ADAPTERS, "mock", _TimeoutAdapter())
    results = asyncio.run(orchestration.orchestrate_run("hi", ["mock"], {}))
    assert results[0].error == "TimeoutError"


def test_nvidia_models_registered_and_free():
    for model in orchestration.NVIDIA_MODELS:
        assert model in orchestration._ADAPTERS
        assert isinstance(orchestration._ADAPTERS[model], NvidiaAdapter)
        assert orchestration.cost_for(model, 1000, 1000) == 0.0


def test_engine_run_with_nvidia_model_and_no_key_records_error_not_crash(monkeypatch):
    monkeypatch.setattr(nvidia_adapter, "NVIDIA_API_KEY", "")
    # AI-only structuring: with no key, the run cannot even structure,
    # so it raises a clear AI-unavailable error instead of running
    # on a hardcoded template. The per-model error behaviour for a
    # successfully structured run is covered elsewhere.
    try:
        asyncio.run(run_engine(
            "Explain cloud computing.", models=["openai/gpt-oss-20b"],
            test_input="cloud computing",
        ))
    except RuntimeError as exc:
        assert "NVIDIA_API_KEY" in str(exc)
        assert "AI structuring unavailable" in str(exc)
    else:
        raise AssertionError("run without a key must fail at AI structuring, not run on rules")
