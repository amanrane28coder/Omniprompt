"""Modules 2-3: preprocessing, intent classification, prompt structuring.

A template uses {{variable}} placeholders. Rendering validates before any
provider call, so a run with a missing variable is rejected, not half-run.
"""
import re

_VARIABLE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_INTENTS = {
    "summarization": ("summar", "summary", "summarise"),
    "code_generation": ("code", "function", "program", "script"),
    "question_answering": ("question", "answer", "explain", "why", "what is"),
}


def clean_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw or "").strip()


def classify_intent(raw: str) -> str:
    text = (raw or "").lower()
    for intent, keywords in _INTENTS.items():
        if any(k in text for k in keywords):
            return intent
    return "general"


def extract_variables(template_text: str) -> dict[str, str]:
    """Variable names in order of first appearance, all typed as string
    unless the caller supplies a richer schema."""
    return {name: "string" for name in _VARIABLE_RE.findall(template_text or "")}


def structure_query(raw: str) -> dict:
    """Turn a raw query into a structured template description."""
    cleaned = clean_text(raw)
    return {
        "objective": cleaned,
        "context": "",
        "constraints": "",
        "format_requirements": "",
        "variables": extract_variables(cleaned),
        "intent": classify_intent(cleaned),
    }


def render_template(template_text: str, variables: dict) -> str:
    missing = [name for name in extract_variables(template_text) if name not in variables]
    if missing:
        raise ValueError(f"Missing variables: {', '.join(missing)}")

    def replace(match: re.Match) -> str:
        return str(variables[match.group(1)])

    return _VARIABLE_RE.sub(replace, template_text)
