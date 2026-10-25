import httpx

from ..config import NVIDIA_API_KEY
from .base import BaseAdapter, CanonicalResponse

BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaAdapter(BaseAdapter):
    """NVIDIA Build hosted API (build.nvidia.com).

    Deliberately OpenAI-compatible: same /chat/completions shape, only
    the base URL, the Bearer key, and the model slug differ. The free
    tier is for prototyping only — rate-limited, not for production —
    and model slugs change when NVIDIA updates versions, so check
    build.nvidia.com for the current slug before pinning a new model.

    This provider is an extension beyond the synopsis, which names
    only OpenAI, Gemini, and Claude. It fits the synopsis design:
    adding a provider touches exactly this one adapter module.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set server-side")
        # 120s, not 30s: NVIDIA's hosted free tier routinely takes
        # longer than 30s to answer, and a 30s cut-off turned live
        # models into empty "successes" (see orchestration timeout fix).
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": params.get("temperature", 0.2),
                    "top_p": params.get("top_p", 1.0),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        choice, usage = data["choices"][0], data.get("usage", {})
        return CanonicalResponse(
            choice["message"]["content"], usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0), choice.get("finish_reason", "stop"),
        )
