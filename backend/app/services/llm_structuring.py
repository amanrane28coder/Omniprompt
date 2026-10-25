"""LLM-based structuring: let the model analyze the raw request.

Rule-based structuring (prompt_engine.structure) is deterministic and
works offline, but it can only assign a role/context/constraint it
has a keyword for. A random user request — "give example of
environment pollution" with no keyword in the list — fell through
to "helpful assistant", and the model then answered generically,
because that is literally the role it was given.

So, when a provider key is available, the model itself analyzes the
request first. gpt-oss-20b (NVIDIA, free) is the structuring model
— the only model on this key that answers structuring calls, as
verified by live testing on Oct 25, 2026. It returns strict JSON;
anything it cannot do — no key, timeout, non-JSON output, missing
fields — is an error, not a fallback to hardcoded rules, because
there is no hardcoded structurer anymore. If the request genuinely
has nothing to understand (a greeting, "Say hi."), the model may
return "helpful assistant", and that is accepted: generic is
correct only there.
"""
import json
import re

from ..adapters.nvidia_adapter import NvidiaAdapter

# The structuring model, chosen by live testing on the user's
# NVIDIA key on Oct 25, 2026, not by a catalog list:
# - openai/gpt-oss-120b: 410 Gone in 2.2s
# - moonshotai/kimi-k3: 410 Gone in 1.0s
# - z-ai/glm-5.3, deepseek-ai/deepseek-v4.1-flash, z-ai/glm-5.3-flash:
#   ReadTimeout at 120s
# - openai/gpt-oss-20b: OK in 16.3s, valid structuring JSON.
# There is no fallback model: if this one cannot judge a request,
# structuring fails with a clear error instead of a hardcoded result.
DEFAULT_STRUCTURING_MODEL = "openai/gpt-oss-20b"

_META_PROMPT = """You analyze a user's raw request and return ONLY a JSON object, no markdown, no explanation, no other text.

Return exactly these fields:
{
  "intent": "what the user wants, in one word or short phrase, e.g. 'explain', 'create', 'analyze', 'summarize' — judge it yourself, do not pick from a fixed list",
  "role": "the role that fits this request's use case — decide it yourself, do not pick from a fixed list and do not limit yourself to these examples. Coding/software usually wants a software engineer, not an instructor. A request that asks for detail/depth usually wants the domain professional (a biologist for detailed biology, a physicist for detailed physics), while a basic teaching/example request usually wants an instructor or tutor. A research request wants an analyst, a writing request a writing coach, and so on — judge by what the user is actually trying to do. The role MUST name the domain: 'biology instructor', never a bare 'instructor', 'teacher', or 'assistant'. Do NOT use 'helpful assistant' unless the request is a greeting or small talk with no topic and no task at all.",
  "objective": "the cleaned request, without any 'Act as ...' prefix",
  "context": "the subject domain or background, e.g. 'Environmental science'. Whenever the request has a topic, this MUST name that domain — never leave it empty for a topical request. Only a greeting/small-talk request may have '' here",
  "audience": "e.g. 'beginner', 'students', or null if not stated",
  "constraints": ["each concrete constraint the user stated, e.g. 'Include one example.', 'Use simple language.' — empty list if none"],
  "format_requirements": "e.g. '5 numbered points', '1 sentence', 'Markdown table', or '' if none stated",
  "format_count": 5,
  "word_limit": 100,
  "gaps": ["each of 'context not specified', 'audience not specified', 'output format not specified', 'word limit not specified' that applies — empty list if all are specified"]
}

Rules:
- format_count is the number in format_requirements when it is numbered points, otherwise null.
- word_limit is the number when the user gave a word limit, otherwise null.
- Do not invent audience, format, or word limit: if the user did not state it, put it in gaps instead.
- Do not invent a specific role for a greeting: 'helpful assistant' is allowed only then.

User request:
"""


def _clean_json(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _normalize(data: dict, raw: str) -> dict | None:
    if not isinstance(data, dict):
        return None
    role = str(data.get("role") or "").strip()
    objective = str(data.get("objective") or "").strip() or raw.strip()
    if not role:
        return None
    constraints = data.get("constraints") or []
    if not isinstance(constraints, list):
        return None
    constraints = [str(c).strip() for c in constraints if str(c).strip()]
    gaps = data.get("gaps") or []
    if not isinstance(gaps, list):
        return None
    gaps = [str(g).strip() for g in gaps if str(g).strip()]
    word_limit = data.get("word_limit")
    if word_limit is not None:
        try:
            word_limit = int(word_limit)
        except (TypeError, ValueError):
            return None
    format_count = data.get("format_count")
    if format_count is not None:
        try:
            format_count = int(format_count)
        except (TypeError, ValueError):
            return None
    audience = data.get("audience")
    audience = str(audience).strip() if audience else None
    intent = data.get("intent")
    intent = str(intent).strip() if intent else "general"
    return {
        "intent": intent,
        "role": role,
        "objective": objective,
        "context": str(data.get("context") or "").strip(),
        "audience": audience,
        "constraints": constraints,
        "format_requirements": str(data.get("format_requirements") or "").strip(),
        "format_count": format_count,
        "word_limit": word_limit,
        "gaps": gaps,
    }


async def llm_structure_with_error(raw: str, model: str = DEFAULT_STRUCTURING_MODEL, adapter=None) -> tuple[dict | None, str | None]:
    """Ask the model to structure `raw`. Returns (dict, None) on
    success, (None, error) on failure — the error is what the
    strict AI path surfaces instead of silently falling back to
    hardcoded rules."""
    if adapter is None:
        # Only NVIDIA models are used for structuring: the mock
        # adapter cannot return JSON, and other providers are not
        # the free structuring model the user chose.
        from .orchestration import NVIDIA_MODELS

        if model not in NVIDIA_MODELS:
            return None, f"model {model} is not a registered structuring model"
        adapter = NvidiaAdapter(model)
    try:
        response = await adapter.send(
            _META_PROMPT + raw, {"temperature": 0.0, "top_p": 1.0}
        )
        data = json.loads(_clean_json(response.output_text))
        normalized = _normalize(data, raw)
        if normalized is None:
            return None, "model returned JSON without a usable role"
        return normalized, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


async def llm_structure(raw: str, model: str = DEFAULT_STRUCTURING_MODEL, adapter=None) -> dict | None:
    """Ask the model to structure `raw`. Returns normalized dict or None.

    None means the model could not structure it. Callers that need
    to know why should use llm_structure_with_error instead.
    """
    result, _ = await llm_structure_with_error(raw, model=model, adapter=adapter)
    return result
