import pytest

from app.services.evaluation import deterministic_evaluate, estimated_cost_usd, weighted_score
from app.services.structuring import extract_variables, render_template, structure_query
from app.services.versioning import branch_template, next_version_no, rollback_template


def test_synopsis_worked_example_scores_82():
    assert weighted_score({"relevance": 4, "adherence": 5, "completeness": 3}) == 82.0


def test_cost_matches_manual_calculation():
    # Synopsis appendix: 812 in / 143 out at gpt-4o-mini placeholder rates
    assert estimated_cost_usd(812, 143, 0.15, 0.60) == round(812 / 1e6 * 0.15 + 143 / 1e6 * 0.60, 6)


def test_template_render_and_missing_variable_rejected():
    template = "Summarize for {{audience}} in under {{max_words}} words."
    assert extract_variables(template) == {"audience": "string", "max_words": "string"}
    assert render_template(template, {"audience": "students", "max_words": 100}) == "Summarize for students in under 100 words."
    with pytest.raises(ValueError, match="max_words"):
        render_template(template, {"audience": "students"})


def test_intent_and_version_numbers():
    assert structure_query("Summarize this article")["intent"] == "summarization"
    assert next_version_no([]) == 1
    assert next_version_no([1, 2]) == 3
    original = "v1 text"
    assert branch_template(original) == original
    assert rollback_template(original) == original  # copy-forward, original untouched
    assert deterministic_evaluate("hello world", "hello world") == deterministic_evaluate("hello world", "hello world")
