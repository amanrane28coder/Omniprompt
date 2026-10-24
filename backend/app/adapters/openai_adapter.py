import httpx

from ..config import OPENAI_API_KEY
from .base import BaseAdapter, CanonicalResponse


class OpenAIAdapter(BaseAdapter):
    model_name = "gpt-4o-mini"

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set server-side")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
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
