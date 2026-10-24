"""Module 7: rubric scoring, cost, latency score, win rate (synopsis §6.10).

Weighted score (0-100) = 100 * sum(weight_i * criterion_i / max_i),
with sum(weight_i) = 1. Worked example from the synopsis: weights
relevance 0.4, adherence 0.35, completeness 0.25, scores 4/5/3 -> 82.0.
"""
import re

DEFAULT_WEIGHTS = {"relevance": 0.4, "adherence": 0.35, "completeness": 0.25}
DEFAULT_RUBRIC_VERSION = "v1"


def weighted_score(scores: dict[str, float], weights: dict[str, float] = DEFAULT_WEIGHTS, max_score: float = 5.0) -> float:
    return round(100 * sum(weights[k] * scores[k] / max_score for k in weights if k in scores), 2)


def estimated_cost_usd(input_tokens: int, output_tokens: int, price_in_per_million: float, price_out_per_million: float) -> float:
    return round((input_tokens / 1e6) * price_in_per_million + (output_tokens / 1e6) * price_out_per_million, 6)


def latency_score(fastest_ms: int, latency_ms: int) -> float:
    if latency_ms <= 0:
        return 0.0
    return round(fastest_ms / latency_ms, 4)


def win_rate(wins: int, losses: int, ties: int) -> float:
    total = wins + losses + ties
    return round(wins / total, 4) if total else 0.0


def _score_0_to_5(value: float) -> int:
    return max(0, min(5, round(value * 5)))


def deterministic_evaluate(output_text: str, test_input: str) -> dict[str, int]:
    """Deterministic placeholder scorer, NOT an LLM judge.

    Relevance: share of test-input keywords present in the output.
    Adherence: output is non-empty and under 2000 words (a stand-in for
    following format constraints until a real judge is wired in).
    Completeness: output length relative to the input, capped.
    Same input always yields the same score (synopsis TC8).
    """
    output = (output_text or "").lower()
    words = set(re.findall(r"[a-z0-9]+", (test_input or "").lower()))
    keywords = {w for w in words if len(w) > 3}
    overlap = len(keywords & set(re.findall(r"[a-z0-9]+", output))) / len(keywords) if keywords else 1.0
    word_count = len(output.split())
    return {
        "relevance": _score_0_to_5(overlap),
        "adherence": 5 if 0 < word_count <= 2000 else 0,
        "completeness": _score_0_to_5(min(1.0, word_count / max(1, len((test_input or '').split()) * 2))),
    }
