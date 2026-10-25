"""Prompt Engine — the core that actually runs (synopsis Modules 2, 3, 5, 7).

Pipeline: raw request -> structured template -> parallel provider run ->
weighted rubric score + latency + cost + format check.

Structuring is deliberately rule-based, not LLM-based: it works offline,
costs $0, and is deterministic, so a demo or test never depends on a
provider key or on a model changing its mind. Real providers enter only
at the run step, through the adapters, under the BYOK model.
"""
import re
from dataclasses import dataclass, field

from .evaluation import deterministic_evaluate, weighted_score
from .orchestration import cost_for, orchestrate_run
from .structuring import clean_text, classify_intent, extract_variables, render_template

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUMBER_RE = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token.lower()]


@dataclass
class StructuredPrompt:
    raw: str
    intent: str
    role: str
    objective: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    format_requirements: str = ""
    format_count: int | None = None
    audience: str | None = None
    word_limit: int | None = None
    variables: dict[str, str] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    template_text: str = ""


def structure(raw: str) -> StructuredPrompt:
    """Turn a raw, vague request into a structured, testable prompt."""
    cleaned = clean_text(raw)
    if not cleaned:
        raise ValueError("Raw request is empty")
    lowered = cleaned.lower()

    # Role: explicit "Act as X" wins, otherwise infer from intent.
    # When a role is given explicitly, drop that sentence from the
    # objective so the template does not repeat "Act as X" twice.
    role_match = re.search(r"act as (?:a |an |the )?([^.,;]+)", lowered)
    objective_text = cleaned
    if role_match:
        role = role_match.group(1).strip()
        objective_text = re.sub(
            r"^act as (?:a |an |the )?[^.,;]+[.,;]\s*", "", cleaned, flags=re.I
        ).strip() or cleaned
    elif "code" in lowered or "function" in lowered or "program" in lowered:
        role = "senior software developer"
    elif "summar" in lowered:
        role = "analyst"
    elif "explain" in lowered or "teach" in lowered:
        role = "subject instructor"
    elif "report" in lowered:
        role = "research analyst"
    else:
        role = "helpful assistant"

    # Audience: "for students", "to a beginner", "for experts", ...
    audience = None
    audience_match = re.search(
        r"(?:for|to) (?:a |an |the )?((?:beginner|beginners|student|students|expert|experts|developer|developers|child|children|professional|professionals)s?)",
        lowered,
    )
    if audience_match:
        audience = audience_match.group(1)

    constraints: list[str] = []
    if audience:
        constraints.append(f"Audience: {audience}.")
    if "practical example" in lowered:
        constraints.append("Include one practical example.")
    if "evidence" in lowered or "evidence-based" in lowered:
        constraints.append("Support claims with evidence.")
    if "step-by-step" in lowered or "step by step" in lowered:
        constraints.append("Explain step by step.")

    word_limit = None
    word_match = re.search(r"(?:under|within|at most|maximum of|max|less than)\s+(\d+)\s+words", lowered)
    if word_match:
        word_limit = int(word_match.group(1))
        constraints.append(f"Keep it under {word_limit} words.")

    format_requirements = ""
    format_count = None
    points_match = re.search(rf"{_NUMBER_RE}\s+numbered points", lowered) or re.search(
        rf"{_NUMBER_RE}\s+points", lowered
    )
    if points_match:
        format_count = _to_int(points_match.group(1))
        format_requirements = f"{format_count} numbered points"
    elif "table" in lowered:
        format_requirements = "Markdown table"
    elif "bullet" in lowered:
        format_requirements = "Bullet list"
    if "two-sentence summary" in lowered or "2-sentence summary" in lowered:
        constraints.append("Finish with a two-sentence summary.")

    gaps: list[str] = []
    if not audience:
        gaps.append("audience not specified")
    if not format_requirements:
        gaps.append("output format not specified")
    if not word_limit:
        gaps.append("word limit not specified")

    sections = [f"Act as {role}.", "", f"Objective: {objective_text}"]
    if constraints:
        sections.append("")
        sections.append("Constraints:")
        sections.extend(f"- {c}" for c in constraints)
    if format_requirements:
        sections.append("")
        sections.append(f"Output format: {format_requirements}.")
    template_text = "\n".join(sections)

    return StructuredPrompt(
        raw=cleaned,
        intent=classify_intent(cleaned),
        role=role,
        objective=objective_text,
        constraints=constraints,
        format_requirements=format_requirements,
        format_count=format_count,
        audience=audience,
        word_limit=word_limit,
        variables=extract_variables(cleaned),
        gaps=gaps,
        template_text=template_text,
    )


def format_check(output_text: str, structured: StructuredPrompt) -> dict:
    """Check the output against the structured format, mechanically.

    This is not a quality judgement — it counts what can be counted
    (numbered points, word count) so a format failure is visible
    instead of hidden inside a subjective score.
    """
    text = output_text or ""
    numbered = len(re.findall(r"(?m)^\s*\d+[.\)]\s", text))
    word_count = len(text.split())
    return {
        "numbered_points_found": numbered,
        "numbered_points_expected": structured.format_count,
        "numbered_points_match": (
            numbered == structured.format_count if structured.format_count else None
        ),
        "word_count": word_count,
        "within_word_limit": (
            word_count <= structured.word_limit if structured.word_limit else None
        ),
    }


async def run_engine(
    raw: str,
    models: list[str] | None = None,
    variables: dict | None = None,
    test_input: str = "",
    temperature: float = 0.2,
    top_p: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Structure a raw request, run it, and score every response.

    Stateless: this does not write to the database. Use the /prompts
    and /runs endpoints when a version, run history, or audit trail
    is wanted. The mock provider needs no key; real providers need
    their key set server-side, and a missing key is recorded as a
    per-model error, not a crash.
    """
    structured = structure(raw)
    prompt_text = render_template(structured.template_text, variables or {})
    if test_input:
        prompt_text = f"{prompt_text}\n\nInput:\n{test_input}"
    params = {"temperature": temperature, "top_p": top_p, "seed": seed}
    results = await orchestrate_run(prompt_text, models or ["mock"], params)
    responses = []
    for result in results:
        scores = None
        weighted = None
        if not result.error:
            scores = deterministic_evaluate(result.response.output_text, test_input or structured.objective)
            weighted = weighted_score(scores)
        responses.append({
            "model": result.model,
            "output_text": result.response.output_text,
            "input_tokens": result.response.input_tokens,
            "output_tokens": result.response.output_tokens,
            "latency_ms": result.latency_ms,
            "finish_reason": result.response.finish_reason,
            "cost_usd": cost_for(result.model, result.response.input_tokens, result.response.output_tokens),
            "error": result.error,
            "scores": scores,
            "weighted_score": weighted,
            "format_check": format_check(result.response.output_text, structured) if not result.error else None,
        })
    return {
        "structured": {
            "intent": structured.intent,
            "role": structured.role,
            "objective": structured.objective,
            "constraints": structured.constraints,
            "format_requirements": structured.format_requirements,
            "format_count": structured.format_count,
            "audience": structured.audience,
            "word_limit": structured.word_limit,
            "variables": structured.variables,
            "gaps": structured.gaps,
            "template_text": structured.template_text,
        },
        "prompt_text": prompt_text,
        "responses": responses,
    }
