import httpx

from ..config import GEMINI_API_KEY
from .base import BaseAdapter, CanonicalResponse


class GeminiAdapter(BaseAdapter):
    model_name = "gemini-2.5-flash"

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set server-side")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url, headers={"x-goog-api-key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt_text}]}]},
            )
            resp.raise_for_status()
            data = resp.json()
        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})
        return CanonicalResponse(
            "".join(p.get("text", "") for p in candidate["content"]["parts"]),
            usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0),
            candidate.get("finishReason", "STOP"),
        )
