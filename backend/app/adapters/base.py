from dataclasses import dataclass


@dataclass
class CanonicalResponse:
    """One shape for every provider (synopsis §6.2)."""
    output_text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


class BaseAdapter:
    model_name = ""

    async def send(self, prompt_text: str, params: dict) -> CanonicalResponse:
        raise NotImplementedError
