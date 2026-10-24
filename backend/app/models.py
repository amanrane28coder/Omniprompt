"""Data model: synopsis Table 6.1, plus two documented extensions.

Extensions: Run.group_id groups the per-model runs of one A/B test, and
ModelResponse.error records a provider failure without crashing the run.
"""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .db import Base


class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    role = Column(String, default="prompt_engineer")


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, default="")
    versions = relationship("PromptVersion", back_populates="prompt")


class PromptVersion(Base):
    """Immutable: there is no update endpoint for this table, by design."""
    __tablename__ = "prompt_versions"
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    parent_version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=True)
    version_no = Column(Integer, nullable=False)
    template_text = Column(Text, nullable=False)
    variables_schema = Column(JSON, default=dict)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    prompt = relationship("Prompt", back_populates="versions")


class TestCase(Base):
    __tablename__ = "test_cases"
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    input_data = Column(JSON, default=dict)
    expected_constraints = Column(String, default="")


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    group_id = Column(String, nullable=False)  # one A/B test = one group
    version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=False)
    model = Column(String, nullable=False)
    temperature = Column(Float, default=0.2)
    top_p = Column(Float, default=1.0)
    seed = Column(Integer, nullable=True)
    status = Column(String, default="completed")
    started_at = Column(DateTime, default=datetime.utcnow)


class ModelResponse(Base):
    __tablename__ = "model_responses"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    model = Column(String, nullable=False)
    output_text = Column(Text, default="")
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    finish_reason = Column(String, default="stop")
    cost_usd = Column(Float, default=0.0)
    error = Column(String, nullable=True)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("model_responses.id"), nullable=False)
    judge_model = Column(String, default="deterministic-placeholder")
    rubric_version = Column(String, default="v1")
    relevance = Column(Integer, default=0)
    adherence = Column(Integer, default=0)
    completeness = Column(Integer, default=0)
    weighted_score = Column(Float, default=0.0)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    entity = Column(String, nullable=False)
    entity_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
