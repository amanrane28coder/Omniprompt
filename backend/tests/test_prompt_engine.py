"""Prompt Engine tests. Structuring is AI-only: these tests fake the
model's JSON (via monkeypatch), never a hardcoded rule list — there
is no rule structurer left to test."""
import asyncio
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_omniprompt.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import llm_structuring as _ls  # noqa: E402
from app.services.prompt_engine import (  # noqa: E402
    StructuredPrompt,
    format_check,
    structure_with_ai_strict,
)

Base.metadata.create_all(bind=engine)
client = TestClient(app)

PPT_EXAMPLE = (
    "Act as a computer science instructor. Explain cloud computing to a "
    "beginner in five numbered points. Include one practical example and "
    "finish with a two-sentence summary."
)

PPT_JSON = {
    "intent": "explain",
    "role": "computer science instructor",
    "objective": "Explain cloud computing to a beginner in five numbered points. Include one practical example and finish with a two-sentence summary.",
    "context": "Computer science",
    "audience": "beginner",
    "constraints": ["Include one practical example.", "Finish with a two-sentence summary."],
    "format_requirements": "5 numbered points",
    "format_count": 5,
    "word_limit": None,
    "gaps": ["word limit not specified"],
}

MAMMALS_JSON = {
    "intent": "explain",
    "role": "biology instructor",
    "objective": "give an example of mammals",
    "context": "Biology",
    "audience": None,
    "constraints": ["Include one example."],
    "format_requirements": "",
    "format_count": None,
    "word_limit": None,
    "gaps": ["audience not specified", "output format not specified", "word limit not specified"],
}


def _fake_ai(data):
    async def fake(raw, model):
        return dict(data), None
    return fake


def test_ppt_example_structures_via_ai():
    # The model assigned everything; the template renderer only
    # turns those fields into sentences, it does not judge them.
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ls, "llm_structure_with_error", _fake_ai(PPT_JSON))
        s, source = asyncio.run(structure_with_ai_strict(PPT_EXAMPLE))
    assert source == "llm:openai/gpt-oss-20b"
    assert s.role == "computer science instructor"
    assert s.audience == "beginner"
    assert s.format_requirements == "5 numbered points"
    assert s.format_count == 5
    assert "Act as computer science instructor." in s.template_text
    # Sentence format, no labels, and no repeats: the objective
    # already says the format, example, summary, and audience, so
    # those sentences are not added a second time.
    assert "Output format:" not in s.template_text
    assert "Context:" not in s.template_text
    assert s.template_text == (
        "Act as computer science instructor. "
        "Explain cloud computing to a beginner in five numbered points. "
        "Include one practical example and finish with a two-sentence summary. "
        "In the context of Computer science."
    )


def test_template_is_sentence_format():
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ls, "llm_structure_with_error", _fake_ai(MAMMALS_JSON))
        s, _ = asyncio.run(structure_with_ai_strict("give an example of mammals"))
    assert s.template_text == (
        "Act as biology instructor. Give an example of mammals. "
        "In the context of Biology. Include one example."
    )


def test_format_check_counts_numbered_points():
    s = StructuredPrompt(
        raw="Explain X in three numbered points.", intent="explain",
        role="instructor", objective="Explain X", format_requirements="3 numbered points",
        format_count=3,
    )
    out = "1. One\n2. Two\n3. Three"
    check = format_check(out, s)
    assert check["numbered_points_found"] == 3
    assert check["numbered_points_match"] is True
    assert format_check("Just a paragraph.", s)["numbered_points_match"] is False


def test_engine_run_endpoint_mock_end_to_end(monkeypatch):
    monkeypatch.setattr(_ls, "llm_structure_with_error", _fake_ai(PPT_JSON))
    res = client.post("/prompt-engine/run", json={
        "raw": PPT_EXAMPLE, "models": ["mock"], "test_input": "cloud computing",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["structured"]["source"] == "llm:openai/gpt-oss-20b"
    assert body["structured"]["format_count"] == 5
    assert body["structured"]["audience"] == "beginner"
    assert len(body["responses"]) == 1
    resp = body["responses"][0]
    assert resp["model"] == "mock"
    assert resp["error"] is None
    assert resp["scores"] is not None
    assert resp["weighted_score"] is not None
    assert resp["format_check"]["numbered_points_expected"] == 5
    # Mock output is not actually numbered, so the format check must
    # honestly say so instead of faking a pass.
    assert resp["format_check"]["numbered_points_match"] is False


def test_engine_structure_endpoint_rejects_empty():
    assert client.post("/prompt-engine/structure", json={"raw": "   "}).status_code == 400


def test_engine_structure_endpoint_ai_failure_is_502_not_rules(monkeypatch):
    async def fake_fail(raw, model):
        return None, "model unavailable"

    monkeypatch.setattr(_ls, "llm_structure_with_error", fake_fail)
    res = client.post("/prompt-engine/structure", json={"raw": "give an example of mammals"})
    assert res.status_code == 502
    assert "AI structuring unavailable" in res.json()["detail"]
    # No rule-based result is hiding in the error response.
    assert "role" not in res.json()
