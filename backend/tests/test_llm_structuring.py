"""LLM structuring tests. No real model call: a fake adapter stands in."""
import asyncio

from app.adapters.base import CanonicalResponse
from app.services import llm_structuring


class _FakeAdapter:
    def __init__(self, output=None, exc=None):
        self.output = output
        self.exc = exc
        self.prompt = None

    async def send(self, prompt_text, params):
        self.prompt = prompt_text
        if self.exc is not None:
            raise self.exc
        return CanonicalResponse(self.output, 0, 0, "stop")


_LLM_JSON = """{
  "role": "environmental science instructor",
  "objective": "give example of environment pollution",
  "context": "Environmental science",
  "audience": null,
  "constraints": ["Include one example."],
  "format_requirements": "",
  "format_count": null,
  "word_limit": null,
  "gaps": ["audience not specified", "output format not specified", "word limit not specified"]
}"""


def test_llm_structure_parses_json_and_assigns_role():
    adapter = _FakeAdapter(_LLM_JSON)
    result = asyncio.run(llm_structuring.llm_structure("give example of environment pollution", adapter=adapter))
    assert result is not None
    assert result["role"] == "environmental science instructor"
    assert result["context"] == "Environmental science"
    assert result["constraints"] == ["Include one example."]


def test_llm_structure_accepts_fenced_json():
    adapter = _FakeAdapter("```json\n" + _LLM_JSON + "\n```")
    result = asyncio.run(llm_structuring.llm_structure("give example of environment pollution", adapter=adapter))
    assert result["role"] == "environmental science instructor"


def test_llm_structure_returns_none_on_bad_output():
    assert asyncio.run(llm_structuring.llm_structure("hi", adapter=_FakeAdapter("not json"))) is None
    assert asyncio.run(llm_structuring.llm_structure("hi", adapter=_FakeAdapter('{"role": ""}'))) is None


def test_llm_structure_returns_none_on_adapter_error_like_missing_key():
    assert asyncio.run(llm_structuring.llm_structure(
        "hi", adapter=_FakeAdapter(exc=RuntimeError("NVIDIA_API_KEY is not set server-side"))
    )) is None


def test_strict_ai_path_uses_model_and_raises_instead_of_rules(monkeypatch):
    import json as _json

    from app.services import llm_structuring as _ls
    from app.services.prompt_engine import structure_with_ai_strict

    async def fake_ok(raw, model):
        return _json.loads(_LLM_JSON), None

    monkeypatch.setattr(_ls, "llm_structure_with_error", fake_ok)
    structured, source = asyncio.run(structure_with_ai_strict("give example of environment pollution"))
    assert source == "llm:openai/gpt-oss-20b"
    assert structured.role == "environmental science instructor"

    async def fake_fail(raw, model):
        return None, f"{model} unavailable"

    monkeypatch.setattr(_ls, "llm_structure_with_error", fake_fail)
    try:
        asyncio.run(structure_with_ai_strict("give example of environment pollution"))
    except RuntimeError as exc:
        assert "gpt-oss-20b" in str(exc)
    else:
        raise AssertionError("strict AI path must raise, not return a hardcoded result")
