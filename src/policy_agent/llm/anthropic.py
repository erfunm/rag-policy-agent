"""Claude adapter: Messages API with native tool use."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from policy_agent.config import LLMSettings
from policy_agent.llm.base import LLMError, LLMTurn, Message, ToolCall, ToolSpec
from policy_agent.models import Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if m.is_error:
                block["is_error"] = True
            # All results for one assistant turn go back in a single user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            if m.raw is not None:
                out.append({"role": "assistant", "content": m.raw})
            else:
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.content})
    return out


_STOP_REASONS = {"end_turn", "tool_use", "max_tokens", "refusal"}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, settings: LLMSettings, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        self.model = settings.anthropic_model
        self._client = client or anthropic.Anthropic(timeout=settings.timeout_s, max_retries=2)

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ],
            # System prompt and tool definitions are stable across requests: cache the prefix.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.effort:
            kwargs["output_config"] = {"effort": self.settings.effort}
        if self.settings.anthropic_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            resp = self._client.beta.messages.create(**kwargs)
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}") from exc

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        u = resp.usage
        input_tokens = (
            u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
        )
        reason = resp.stop_reason or "other"
        return LLMTurn(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=input_tokens, output_tokens=u.output_tokens),
            stop_reason=reason if reason in _STOP_REASONS else "other",
            model=resp.model,
            raw=resp.content,
        )
