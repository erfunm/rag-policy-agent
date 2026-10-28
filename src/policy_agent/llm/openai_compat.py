"""OpenAI-compatible chat.completions adapter. Also covers local servers (Ollama, vLLM) via base_url."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import openai

from policy_agent.config import LLMSettings
from policy_agent.llm.base import LLMError, LLMTurn, Message, ToolCall, ToolSpec
from policy_agent.models import Usage

_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


def to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        else:
            out.append({"role": "user", "content": m.content})
    return out


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        # Surface to the dispatcher, which reports a validation error back to the model.
        return {"__unparseable__": raw}
    return parsed if isinstance(parsed, dict) else {"__unparseable__": raw}


class OpenAICompatClient:
    provider = "openai"

    def __init__(self, settings: LLMSettings, client: openai.OpenAI | None = None) -> None:
        self.settings = settings
        self.model = settings.openai_model
        self._client = client or openai.OpenAI(
            base_url=settings.openai_base_url,
            # Local OpenAI-compatible servers usually ignore the key but the SDK requires one.
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
            timeout=settings.timeout_s,
            max_retries=2,
        )

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                messages=to_openai_messages(system, messages),  # type: ignore[arg-type]
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                    for t in tools
                ],
            )
        except openai.APIConnectionError as exc:
            raise LLMError(f"openai connection error: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"openai API error {exc.status_code}: {exc.message}") from exc

        choice = resp.choices[0]
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=_parse_arguments(c.function.arguments))
            for c in (choice.message.tool_calls or [])
            if c.type == "function"
        ]
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return LLMTurn(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=usage,
            stop_reason=_STOP.get(choice.finish_reason or "", "other"),
            model=resp.model,
        )
