"""The contract every LLM adapter implements: one chat turn with native tool calling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from policy_agent.models import Usage


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """Provider-neutral transcript entry. Adapters translate to and from their wire format."""

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False
    # Provider-native assistant content, replayed verbatim so e.g. Claude thinking blocks stay intact.
    raw: Any = None


class LLMTurn(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    # Normalised: end_turn | tool_use | max_tokens | refusal | other
    stop_reason: str = "end_turn"
    model: str = ""
    raw: Any = None

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls, raw=self.raw)


class LLMError(RuntimeError):
    """Raised by adapters for provider failures the agent should turn into an escalation."""


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn: ...
