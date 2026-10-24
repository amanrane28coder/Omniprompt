from .base import BaseAdapter, CanonicalResponse


class MockAdapter(BaseAdapter):
    """Offline provider for demos, tests and rate-limit-safe demo runs."""

    model_name = "mock"

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        text = f"[mock] Response to: {prompt_text}"
        return CanonicalResponse(text, len(prompt_text) // 4, len(text) // 4, "stop")
