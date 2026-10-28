"""Tool registry: JSON-schema specs from pydantic models, validated dispatch with timeouts.

Every failure (unknown tool, invalid arguments, exception, timeout) becomes an error tool result
that goes back to the model. A broken tool never crashes the agent loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from policy_agent.config import Settings
from policy_agent.llm.base import ToolCall, ToolSpec
from policy_agent.models import Role, ToolCallRecord

if TYPE_CHECKING:
    from policy_agent.retrieval.hybrid import HybridRetriever
    from policy_agent.tools.cases import CaseStore


@dataclass(frozen=True)
class ToolContext:
    """Per-request context. Role and date come from the authenticated request, never from the model."""

    role: Role
    as_of: date
    retriever: HybridRetriever
    cases: CaseStore
    settings: Settings


class ToolOutput(BaseModel):
    content: dict[str, Any]
    evidence_ids: list[str] = Field(default_factory=list)
    is_error: bool = False


def error_output(message: str) -> ToolOutput:
    return ToolOutput(content={"error": message}, is_error=True)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], ToolOutput]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


def _format_validation(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool], max_workers: int = 4) -> None:
        self._tools = {t.name: t for t in tools}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def execute(
        self, call: ToolCall, ctx: ToolContext, timeout_s: float
    ) -> tuple[ToolOutput, ToolCallRecord]:
        start = time.perf_counter()
        output = self._run(call, ctx, timeout_s)
        record = ToolCallRecord(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            is_error=output.is_error,
            duration_ms=(time.perf_counter() - start) * 1000,
            evidence_ids=output.evidence_ids,
        )
        return output, record

    def _run(self, call: ToolCall, ctx: ToolContext, timeout_s: float) -> ToolOutput:
        tool = self._tools.get(call.name)
        if tool is None:
            return error_output(f"unknown tool {call.name!r}; available: {', '.join(self._tools)}")
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return error_output(f"invalid arguments: {_format_validation(exc)}")
        future = self._pool.submit(tool.fn, args, ctx)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout:
            future.cancel()
            return error_output(f"tool {call.name} timed out after {timeout_s:.1f}s")
        except Exception as exc:
            return error_output(f"tool {call.name} failed: {type(exc).__name__}: {exc}")
