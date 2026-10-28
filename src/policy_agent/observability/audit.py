"""Append-only JSONL audit log. Every free-text field is PII-redacted, whatever the input setting."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policy_agent.guardrails import pii
from policy_agent.models import AgentResponse, AskRequest


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return pii.redact(value).text
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        line = json.dumps(_redact_value(event), ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_request(
        self, request: AskRequest, response: AgentResponse, provider: str, trace_id: str | None
    ) -> None:
        self.write(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "agent_request",
                "request_id": response.request_id,
                "trace_id": trace_id,
                "role": request.role,
                "as_of": request.as_of,
                "question": request.question,
                "prompt_version": response.prompt_version,
                "provider": provider,
                "model": response.model,
                "status": response.status,
                "reason": response.reason,
                "confidence": response.confidence,
                "flags": response.flags,
                "citations": response.citations,
                "retrieved_ids": response.retrieved_ids,
                "tool_calls": [t.model_dump() for t in response.tool_calls],
                "answer": response.answer,
                "usage": response.usage.model_dump(),
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "steps": response.steps,
            }
        )
