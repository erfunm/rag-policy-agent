"""Domain models shared across ingestion, retrieval, tools, the agent loop and the API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["support", "operations", "compliance"]
Status = Literal["answered", "escalated"]
Confidence = Literal["high", "medium", "low"]

ROLES: tuple[Role, ...] = ("support", "operations", "compliance")


class _Versioned(BaseModel):
    """Fields that decide whether a policy (or a chunk of one) applies to a request."""

    policy_id: str
    version: str
    title: str
    effective_date: date
    # Set by the loader when a later version of the same policy takes effect.
    superseded_on: date | None = None
    acl_roles: list[Role]

    def in_force(self, as_of: date) -> bool:
        return self.effective_date <= as_of and (self.superseded_on is None or as_of < self.superseded_on)

    def visible_to(self, role: Role) -> bool:
        return role in self.acl_roles

    @property
    def key(self) -> str:
        return f"{self.policy_id}@{self.version}"


class PolicyDoc(_Versioned):
    owner: str
    summary: str
    body: str
    source: str


class Chunk(_Versioned):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    section: str
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    rank: int
    score: float
    # Dense cosine similarity to the query; comparable across queries, used for the evidence floor.
    relevance: float


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ToolCallRecord(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    is_error: bool
    duration_ms: float
    evidence_ids: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    role: Role = "support"
    as_of: date | None = None


class AgentResponse(BaseModel):
    request_id: str
    status: Status
    answer: str
    citations: list[str]
    confidence: Confidence
    reason: str | None = None
    flags: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    steps: int = 0
    model: str
    prompt_version: str
