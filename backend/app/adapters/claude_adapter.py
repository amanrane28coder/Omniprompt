import httpx

from ..config import ANTHROPIC_API_KEY
from .base import BaseAdapter, CanonicalResponse


class ClaudeAdapter(BaseAdapter):
    model_name = "claude-sonnet-4-5"

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set server-side")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model_name, "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": params.get("temperature", 0.2),
                    "top_p": params.get("top_p", 1.0),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage", {})
        return CanonicalResponse(
            "".join(b.get("text", "") for b in data.get("content", [])),
            usage.get("input_tokens", 0), usage.get("output_tokens", 0),
            data.get("stop_reason", "end_turn"),
        )
