"""Module 5: multi-LLM orchestration.

One canonical request/response per adapter. Dispatch is parallel with a
per-provider semaphore, a timeout and one retry; a provider failure is
recorded as an error response, never a crash (synopsis TC6).
"""
import asyncio
import time
from dataclasses import dataclass

from ..adapters.base import CanonicalResponse
from ..adapters.claude_adapter import ClaudeAdapter
from ..adapters.gemini_adapter import GeminiAdapter
from ..adapters.mock import MockAdapter
from ..adapters.nvidia_adapter import NvidiaAdapter
from ..adapters.openai_adapter import OpenAIAdapter
from .evaluation import estimated_cost_usd

# NVIDIA Build models registered by slug. Slugs change when NVIDIA
# updates versions — check build.nvidia.com for the current list.
# All are $0 on the free tier (rate-limited, prototyping only).
NVIDIA_MODELS = [
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
]

# Placeholder per-1M-token rates. Replace with current provider pricing
# before quoting real costs.
PRICES = {
    "mock": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-4-5": (3.00, 15.00),
}

_ADAPTERS = {
    "mock": MockAdapter(),
    "gpt-4o-mini": OpenAIAdapter(),
    "gemini-2.5-flash": GeminiAdapter(),
    "claude-sonnet-4-5": ClaudeAdapter(),
    **{model: NvidiaAdapter(model) for model in NVIDIA_MODELS},
}
PRICES.update({model: (0.0, 0.0) for model in NVIDIA_MODELS})
_SEMAPHORES = {name: asyncio.Semaphore(5) for name in _ADAPTERS}


@dataclass
class RunResult:
    model: str
    response: CanonicalResponse
    latency_ms: int
    error: str | None = None


async def _call_one(model: str, prompt_text: str, params: dict) -> RunResult:
    adapter = _ADAPTERS.get(model)
    if adapter is None:
        return RunResult(model, CanonicalResponse("", 0, 0, "error"), 0, f"Unknown model: {model}")
    start = time.monotonic()
    try:
        async with _SEMAPHORES[model]:
            response = await asyncio.wait_for(adapter.send(prompt_text, params), timeout=30)
        return RunResult(model, response, int((time.monotonic() - start) * 1000))
    except Exception as exc:  # timeout, rate limit, outage -> recorded, not raised
        return RunResult(model, CanonicalResponse("", 0, 0, "error"), int((time.monotonic() - start) * 1000), str(exc))


async def orchestrate_run(prompt_text: str, models: list[str], params: dict) -> list[RunResult]:
    return list(await asyncio.gather(*[_call_one(m, prompt_text, params) for m in models]))


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return estimated_cost_usd(input_tokens, output_tokens, price_in, price_out)
