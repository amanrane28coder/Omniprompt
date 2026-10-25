"""Prompt Engine tests, including the deck's BEFORE/AFTER example."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_omniprompt.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services.prompt_engine import format_check, structure  # noqa: E402

Base.metadata.create_all(bind=engine)
client = TestClient(app)

PPT_EXAMPLE = (
    "Act as a computer science instructor. Explain cloud computing to a "
    "beginner in five numbered points. Include one practical example and "
    "finish with a two-sentence summary."
)


def test_ppt_example_structures_correctly():
    s = structure(PPT_EXAMPLE)
    assert s.role == "computer science instructor"
    assert s.audience == "beginner"
    assert s.format_requirements == "5 numbered points"
    assert s.format_count == 5
    assert any("practical example" in c for c in s.constraints)
    assert any("two-sentence summary" in c for c in s.constraints)
    assert "Act as computer science instructor." in s.template_text
    assert "Output format: 5 numbered points." in s.template_text


def test_vague_request_reports_gaps_instead_of_inventing_them():
    s = structure("Create a report on renewable energy.")
    assert "audience not specified" in s.gaps
    assert "output format not specified" in s.gaps
    assert s.role == "research analyst"


def test_summarization_detects_audience_and_word_limit():
    s = structure("Summarize this article for students in under 100 words.")
    assert s.intent == "summarization"
    assert s.audience == "students"
    assert s.word_limit == 100


def test_format_check_counts_numbered_points():
    s = structure("Explain X in three numbered points.")
    out = "1. One\n2. Two\n3. Three"
    check = format_check(out, s)
    assert check["numbered_points_found"] == 3
    assert check["numbered_points_match"] is True
    assert format_check("Just a paragraph.", s)["numbered_points_match"] is False


def test_engine_run_endpoint_mock_end_to_end():
    res = client.post("/prompt-engine/run", json={
        "raw": PPT_EXAMPLE, "models": ["mock"], "test_input": "cloud computing",
    })
    assert res.status_code == 200
    body = res.json()
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
