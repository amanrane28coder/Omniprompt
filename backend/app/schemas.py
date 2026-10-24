from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str


class PromptCreate(BaseModel):
    title: str
    description: str = ""
    template_text: str


class VersionCreate(BaseModel):
    template_text: str
    parent_version_id: Optional[int] = None


class RunCreate(BaseModel):
    version_id: int
    models: list[str] = Field(default=["mock"])
    variables: dict[str, Any] = {}
    test_input: str = ""
    temperature: float = 0.2
    top_p: float = 1.0
    seed: Optional[int] = None
