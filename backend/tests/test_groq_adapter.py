"""Groq fast-path tests. No live Groq key is used here: the adapter
is tested through its slug map, its missing-key error, and the
run-structured endpoint with the mock provider."""
import asyncio
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_omniprompt.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.adapters.groq_adapter import API_SLUGS, GroqAdapter  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import orchestration as orch  # noqa: E402

Base.metadata.create_all(bind=engine)
client = TestClient(app)


def test_groq_slug_map():
    assert API_SLUGS["groq/gpt-oss-20b"] == "openai/gpt-oss-20b"
    assert API_SLUGS["groq/gpt-oss-120b"] == "openai/gpt-oss-120b"
    assert API_SLUGS["groq/llama-3.3-70b-versatile"] == "llama-3.3-70b-versatile"
    assert GroqAdapter("groq/gpt-oss-120b").api_model == "openai/gpt-oss-120b"


def test_groq_models_registered_in_orchestration():
    for model in orch.GROQ_MODELS:
        assert isinstance(orch._ADAPTERS[model], GroqAdapter)
        assert model not in orch.NVIDIA_MODELS


def test_groq_missing_key_is_error_not_crash():
    import app.adapters.groq_adapter as ga

    old = ga.GROQ_API_KEY
    ga.GROQ_API_KEY = ""
    try:
        with __import__("pytest").raises(RuntimeError, match="GROQ_API_KEY"):
            asyncio.run(GroqAdapter("groq/gpt-oss-20b").send("hi", {}))
    finally:
        ga.GROQ_API_KEY = old


def test_run_structured_reuses_template_without_structuring():
    # No llm call is involved: run-structured must work with the
    # mock provider even if structuring would fail.
    structured = {
        "source": "llm:openai/gpt-oss-20b",
        "intent": "explain",
        "role": "biology instructor",
        "objective": "give an example of mammals",
        "context": "Biology",
        "constraints": ["Include one example."],
        "format_requirements": "",
        "format_count": None,
        "audience": None,
        "word_limit": None,
        "variables": {},
        "gaps": [],
        "template_text": "Act as biology instructor. Give an example of mammals. In the context of Biology. Include one example.",
    }
    res = client.post("/prompt-engine/run-structured", json={
        "structured": structured, "models": ["mock"], "test_input": "mammals",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["structured"]["template_text"] == structured["template_text"]
    assert body["responses"][0]["model"] == "mock"
    assert body["responses"][0]["error"] is None


def test_run_structured_rejects_empty_template():
    res = client.post("/prompt-engine/run-structured", json={
        "structured": {"template_text": ""}, "models": ["mock"],
    })
    assert res.status_code == 400
