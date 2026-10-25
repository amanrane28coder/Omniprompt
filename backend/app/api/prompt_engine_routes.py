"""Prompt Engine endpoints: structure a raw request, or structure + run it.

Stateless by design — no database writes here. For versioned prompts,
run history, and audit events, use /prompts and /runs instead.
"""
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.prompt_engine import run_engine, structure_with_ai_strict

router = APIRouter(prefix="/prompt-engine", tags=["prompt-engine"])


class StructureRequest(BaseModel):
    raw: str
    # No use_llm flag anymore: structuring is AI-only
    # (gpt-oss-20b, live). If the model cannot judge the request,
    # the endpoint returns 502 with the model error — there is
    # no hardcoded structurer to fall back to.


class RunRequest(BaseModel):
    raw: str
    models: list[str] = ["mock"]
    variables: dict[str, Any] = {}
    test_input: str = ""
    temperature: float = 0.2
    top_p: float = 1.0
    seed: Optional[int] = None


@router.post("/structure")
async def structure_endpoint(payload: StructureRequest):
    try:
        s, source = await structure_with_ai_strict(payload.raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    return {
        "source": source,
        "intent": s.intent, "role": s.role, "objective": s.objective,
        "context": s.context,
        "constraints": s.constraints, "format_requirements": s.format_requirements,
        "format_count": s.format_count, "audience": s.audience,
        "word_limit": s.word_limit, "variables": s.variables, "gaps": s.gaps,
        "template_text": s.template_text,
    }


@router.post("/run")
async def run_endpoint(payload: RunRequest):
    try:
        return await run_engine(
            payload.raw, payload.models, payload.variables, payload.test_input,
            payload.temperature, payload.top_p, payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
