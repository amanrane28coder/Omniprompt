import httpx

from ..config import GROQ_API_KEY
from .base import BaseAdapter, CanonicalResponse

BASE_URL = "https://api.groq.com/openai/v1"

# Display name (used in the UI / API) -> real Groq API slug.
# Groq's gpt-oss slugs carry the "openai/" prefix on their API,
# while the llama slug does not — hence the explicit map instead
# of stripping a prefix.
API_SLUGS = {
    "groq/gpt-oss-20b": "openai/gpt-oss-20b",
    "groq/gpt-oss-120b": "openai/gpt-oss-120b",
    "groq/llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
}


class GroqAdapter(BaseAdapter):
    """Groq hosted API (groq.com).

    Deliberately OpenAI-compatible, like the NVIDIA adapter: same
    /chat/completions shape, only the base URL, the Bearer key, and
    the model slug differ. Groq is the fast path: its inference is
    typically an order of magnitude quicker than NVIDIA's free tier,
    which is what makes a real run feasible in a live demo. Keys are
    free-tier (rate-limited); set GROQ_API_KEY server-side, never
    in the repo. If the key is absent, this raises like every other
    adapter with a missing key — it is never silently skipped.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_model = API_SLUGS.get(model_name, model_name)

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set server-side")
        # 30s, not 120s: Groq is fast by design, and a 30s cut-off
        # keeps a slow viva demo from hanging for two minutes.
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": self.api_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": params.get("temperature", 0.2),
                "top_p": params.get("top_p", 1.0),
            }
            if params.get("max_tokens") is not None:
                payload["max_tokens"] = params["max_tokens"]
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        choice, usage = data["choices"][0], data.get("usage", {})
        return CanonicalResponse(
            choice["message"]["content"], usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0), choice.get("finish_reason", "stop"),
        )
