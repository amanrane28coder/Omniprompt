"""Prompt Engine — the core that actually runs (synopsis Modules 2, 3, 5, 7).

Pipeline: raw request -> structured template -> parallel provider run ->
weighted rubric score + latency + cost + format check.

Structuring is AI-only, by design: a live model (gpt-oss-20b on
NVIDIA, the only model on this key that actually answers structuring
calls as of Oct 25, 2026 — gpt-oss-120b and kimi-k3 return 410 Gone,
glm/deepseek time out at 120s) judges the request and assigns role,
context, constraints, and output format. There is no hardcoded
rule-based structurer in this module anymore: if the model cannot
judge a request, structuring fails with a clear error instead of
returning a keyword-based result. Real providers enter at the run
step through the adapters, under the BYOK model.
"""
import re
from dataclasses import dataclass, field

from .evaluation import deterministic_evaluate, weighted_score
from .orchestration import cost_for, orchestrate_run
from .structuring import clean_text, extract_variables, render_template


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


def _as_sentence(text: str) -> str:
    """One sentence, capitalized, ending with a period — unless it
    already ends with '?' or '!', in which case no '.' is added
    ('What is python programming?.' is not a sentence format)."""
    text = text.strip()
    if not text:
        return ""
    capitalized = text[0].upper() + text[1:]
    if capitalized[-1] in "?!":
        return capitalized
    return capitalized.rstrip(".") + "."


def _format_sentence(format_requirements: str) -> str:
    """Format field as a sentence, not a label: '5 numbered points'
    -> 'Use 5 numbered points.', 'Markdown table' -> 'Present it
    as a Markdown table.' — the field itself stays machine-readable
    for format_check; only the template phrasing changes."""
    fmt = format_requirements.strip()
    if fmt == "Markdown table":
        return "Present it as a Markdown table."
    if fmt == "Numbered list":
        return "Use a numbered list."
    if fmt == "Bullet list":
        return "Use a bullet list."
    if fmt == "Paragraphs":
        return "Write it in paragraphs."
    if fmt == "Essay":
        return "Write it as an essay."
    if fmt == "JSON":
        return "Return it as JSON."
    if fmt == "Markdown":
        return "Write it in Markdown."
    if re.match(r"^\d+ sentences?$", fmt):
        return f"Write it in {fmt}."
    return f"Use {fmt}."


_NUMBER_WORD_DIGITS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _normalize_numbers(text: str) -> str:
    lowered = text.lower()
    for word, digit in _NUMBER_WORD_DIGITS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    return lowered


def _build_template(role: str, objective: str, context: str, constraints: list[str], format_requirements: str) -> str:
    """Sentence-format template, like the cloud computing example:
    'Act as a computer science instructor. Explain cloud computing
    to a beginner in five numbered points. Include one practical
    example and finish with a two-sentence summary.' — one natural
    paragraph, no 'Objective:' / 'Context:' / 'Output format:'
    labels. This is the text sent to the model, so it reads like
    a prompt a person would write, not a form."""
    sentences = [f"Act as {role}.", _as_sentence(objective)]
    if context:
        sentences.append(f"In the context of {context.rstrip('.')}.")
    objective_lower = objective.lower()
    for constraint in constraints:
        # "Audience: beginner." is a label, not a sentence — and it
        # is dropped entirely when the objective already names the
        # audience ("to a beginner"), otherwise the sentence format
        # would repeat what the objective just said.
        audience_match = re.match(r"^Audience:\s*(.+)\.$", constraint.strip())
        if audience_match:
            if audience_match.group(1).lower() not in objective_lower:
                sentences.append(f"The audience is {audience_match.group(1)}.")
            continue
        # Same dedup for constraints the objective already states
        # verbatim ("Include one practical example.").
        if _as_sentence(constraint).lower().rstrip(".") not in objective_lower:
            sentences.append(_as_sentence(constraint))
    if format_requirements:
        # "five numbered points" in the objective is the same format
        # as "5 numbered points" — normalize words to digits first.
        if format_requirements.lower() not in _normalize_numbers(objective):
            sentences.append(_format_sentence(format_requirements))
    return " ".join(s for s in sentences if s)


async def structure_with_ai_strict(raw: str) -> tuple[StructuredPrompt, str]:
    """Structure with AI only — no hardcoded structurer exists
    anymore to fall back to.

    The model (gpt-oss-20b, DEFAULT_STRUCTURING_MODEL) judges the
    request and assigns role, context, constraints, format, and
    intent. If it cannot judge the request, this raises
    RuntimeError with the model error instead of returning any
    keyword-based result.
    """
    from .llm_structuring import DEFAULT_STRUCTURING_MODEL, llm_structure_with_error

    cleaned = clean_text(raw)
    if not cleaned:
        raise ValueError("Raw request is empty")
    llm, error = await llm_structure_with_error(cleaned, model=DEFAULT_STRUCTURING_MODEL)
    if llm is None:
        raise RuntimeError(
            f"AI structuring unavailable — {DEFAULT_STRUCTURING_MODEL}: {error}. "
            "No other model on this NVIDIA key answers structuring calls: "
            "gpt-oss-120b and kimi-k3 return 410 Gone, glm/deepseek time out "
            "at 120s (verified Oct 25, 2026)."
        )
    template_text = _build_template(
        llm["role"], llm["objective"], llm["context"],
        llm["constraints"], llm["format_requirements"],
    )
    return StructuredPrompt(
        raw=cleaned,
        intent=llm.get("intent") or "general",
        role=llm["role"],
        objective=llm["objective"],
        context=llm["context"],
        constraints=llm["constraints"],
        format_requirements=llm["format_requirements"],
        format_count=llm["format_count"],
        audience=llm["audience"],
        word_limit=llm["word_limit"],
        variables=extract_variables(cleaned),
        gaps=llm["gaps"],
        template_text=template_text,
    ), f"llm:{DEFAULT_STRUCTURING_MODEL}"


def format_check(output_text: str, structured: StructuredPrompt) -> dict:
    """Check the output against the structured format, mechanically.

    This is not a quality judgement — it counts what can be counted
    (numbered points, word count) so a format failure is visible
    instead of hidden inside a subjective score.
    """
    text = output_text or ""
    numbered = len(re.findall(r"(?m)^\s*\d+[.\)]\s", text))
    word_count = len(text.split())
    result = {
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
    # The formats structure() can now assign are also checked here,
    # mechanically, instead of being assigned and then ignored.
    fmt = structured.format_requirements or ""
    sentence_match = re.match(r"(\d+) sentences?$", fmt)
    if sentence_match:
        expected = int(sentence_match.group(1))
        found = len([s for s in re.split(r"[.!?]+\s*", text.strip()) if s.strip()])
        result["sentences_found"] = found
        result["sentences_expected"] = expected
        result["sentences_match"] = found == expected
    if fmt == "Bullet list":
        bullets = len(re.findall(r"(?m)^\s*[-*•]\s", text))
        result["bullets_found"] = bullets
        result["bullets_match"] = bullets > 0
    return result


async def run_engine(
    raw: str,
    models: list[str] | None = None,
    variables: dict | None = None,
    test_input: str = "",
    temperature: float = 0.2,
    top_p: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Structure a raw request with AI, run it, and score every response.

    Stateless: this does not write to the database. Use the /prompts
    and /runs endpoints when a version, run history, or audit trail
    is wanted. The mock provider needs no key; real providers need
    their key set server-side, and a missing key is recorded as a
    per-model error, not a crash. Structuring itself is AI-only:
    if the structuring model cannot judge the request, this raises
    instead of running on a hardcoded template.
    """
    structured, source = await structure_with_ai_strict(raw)
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
            "source": source,
            "intent": structured.intent,
            "role": structured.role,
            "objective": structured.objective,
            "context": structured.context,
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
