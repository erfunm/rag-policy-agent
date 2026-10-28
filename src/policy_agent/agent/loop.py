"""The agent loop, written by hand on purpose.

    question -> LLM -> tool calls -> execute -> results back to the LLM -> ... -> submit_answer

The LLM decides which tools to call. The loop owns everything that must not depend on the model
behaving: the step limit, tool timeouts, turning tool errors into tool results, validating the
final structured answer, and downgrading to an escalation when it cannot be trusted.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from collections.abc import Generator, Iterator
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from policy_agent.agent.prompts import (
    NUDGE_SUBMIT,
    PROMPT_VERSION,
    SUBMIT_ANSWER_DESCRIPTION,
    SYSTEM_PROMPT,
    render_user_message,
)
from policy_agent.config import Settings
from policy_agent.guardrails import grounding, injection, pii
from policy_agent.llm.base import LLMClient, LLMError, LLMTurn, Message, ToolCall, ToolSpec
from policy_agent.models import AgentResponse, AskRequest, Confidence, Status, ToolCallRecord, Usage
from policy_agent.observability.audit import AuditLogger
from policy_agent.observability.tracing import Span, Tracer, setup_tracing
from policy_agent.retrieval.hybrid import HybridRetriever
from policy_agent.tools import ToolContext, ToolRegistry, default_registry
from policy_agent.tools.cases import CaseStore

SUBMIT_ANSWER = "submit_answer"


class SubmitAnswerArgs(BaseModel):
    status: Status
    answer: str = Field(description="Answer with inline [chunk_id] citations. Ignored when escalating.")
    citations: list[str] = Field(default_factory=list, description="Every evidence id cited in the answer.")
    confidence: Confidence = "medium"
    reason: str | None = Field(default=None, description="Why the question is being escalated.")


SUBMIT_SPEC = ToolSpec(
    name=SUBMIT_ANSWER,
    description=SUBMIT_ANSWER_DESCRIPTION,
    parameters={k: v for k, v in SubmitAnswerArgs.model_json_schema().items() if k != "title"},
)


class AgentEvent(BaseModel):
    type: Literal["start", "llm_turn", "tool_call", "tool_result", "final"]
    data: dict[str, Any] = Field(default_factory=dict)


class _RunState:
    def __init__(self, request: AskRequest, request_id: str) -> None:
        self.request = request
        self.request_id = request_id
        self.started = time.perf_counter()
        self.usage = Usage()
        self.steps = 0
        self.flags: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.evidence: list[str] = []
        self.model = ""


class PolicyAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        retriever: HybridRetriever,
        cases: CaseStore,
        registry: ToolRegistry | None = None,
        audit: AuditLogger | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.cases = cases
        self.registry = registry or default_registry()
        self.audit = audit or AuditLogger(settings.observability.audit_log_path)
        self.tracer = tracer or Tracer()
        self.tool_specs = [*self.registry.specs(), SUBMIT_SPEC]

    def ask(self, request: AskRequest) -> AgentResponse:
        final: AgentResponse | None = None
        for event in self.stream(request):
            if event.type == "final":
                final = AgentResponse.model_validate(event.data)
        assert final is not None
        return final

    def stream(self, request: AskRequest) -> Iterator[AgentEvent]:
        state = _RunState(request, uuid.uuid4().hex)
        as_of = request.as_of or date.today()
        root = self.tracer.start(
            "invoke_agent policy-agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "policy-agent",
                "gen_ai.provider.name": self.llm.provider,
                "gen_ai.request.model": self.llm.model,
                "gen_ai.prompt.version": PROMPT_VERSION,
                "policy_agent.request_id": state.request_id,
                "policy_agent.role": request.role,
                "policy_agent.as_of": as_of.isoformat(),
            },
        )
        try:
            yield AgentEvent(
                type="start",
                data={"request_id": state.request_id, "as_of": as_of.isoformat(), "trace_id": root.trace_id},
            )
            ctx = ToolContext(
                role=request.role,
                as_of=as_of,
                retriever=self.retriever,
                cases=self.cases,
                settings=self.settings,
                tracer=self.tracer,
            )
            question = self._screen_input(state)
            messages = [Message(role="user", content=render_user_message(question, as_of))]
            response = yield from self._run(state, ctx, messages, root)
            root.set_many(
                {
                    "gen_ai.response.model": response.model,
                    "gen_ai.usage.input_tokens": response.usage.input_tokens,
                    "gen_ai.usage.output_tokens": response.usage.output_tokens,
                    "policy_agent.status": response.status,
                    "policy_agent.reason": response.reason,
                    "policy_agent.flags": response.flags,
                    "policy_agent.citations": response.citations,
                    "policy_agent.steps": response.steps,
                    "policy_agent.cost_usd": response.cost_usd,
                }
            )
            self.audit.log_request(request, response, self.llm.provider, root.trace_id)
            yield AgentEvent(type="final", data=response.model_dump(mode="json"))
        finally:
            root.end()

    def _chat(self, messages: list[Message], parent: Span) -> LLMTurn:
        span = self.tracer.start(
            f"chat {self.llm.model}",
            kind="client",
            parent=parent,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": self.llm.provider,
                "gen_ai.request.model": self.llm.model,
                "gen_ai.request.max_tokens": self.settings.llm.max_tokens,
            },
        )
        try:
            turn = self.llm.chat(SYSTEM_PROMPT, messages, self.tool_specs)
        except LLMError as exc:
            span.error(type(exc).__name__, str(exc))
            span.end()
            raise
        span.set_many(
            {
                "gen_ai.response.model": turn.model or None,
                "gen_ai.response.finish_reasons": [turn.stop_reason],
                "gen_ai.usage.input_tokens": turn.usage.input_tokens,
                "gen_ai.usage.output_tokens": turn.usage.output_tokens,
            }
        )
        span.end()
        return turn

    def _run(
        self, state: _RunState, ctx: ToolContext, messages: list[Message], root: Span
    ) -> Generator[AgentEvent, None, AgentResponse]:
        nudged = False
        for step in range(1, self.settings.agent.max_steps + 1):
            state.steps = step
            try:
                turn = self._chat(messages, root)
            except LLMError as exc:
                return self._escalate(state, f"llm_error: {exc}")
            state.usage += turn.usage
            state.model = turn.model or self.llm.model
            yield AgentEvent(
                type="llm_turn",
                data={
                    "step": step,
                    "stop_reason": turn.stop_reason,
                    "tool_calls": [c.name for c in turn.tool_calls],
                },
            )
            if turn.stop_reason == "refusal":
                return self._escalate(state, "model_refusal")
            messages.append(turn.as_message())

            if not turn.tool_calls:
                if nudged:
                    return self._escalate(state, "no_structured_answer")
                nudged = True
                messages.append(Message(role="user", content=NUDGE_SUBMIT))
                continue

            submit = next((c for c in turn.tool_calls if c.name == SUBMIT_ANSWER), None)
            for call in turn.tool_calls:
                if call.name == SUBMIT_ANSWER:
                    continue
                yield AgentEvent(
                    type="tool_call", data={"id": call.id, "name": call.name, "arguments": call.arguments}
                )
                tool_span = self.tracer.start(
                    f"execute_tool {call.name}",
                    parent=root,
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": call.name,
                        "gen_ai.tool.call.id": call.id,
                        "gen_ai.tool.type": "function",
                    },
                )
                output, record = self.registry.execute(
                    call, dataclasses.replace(ctx, span=tool_span), self.settings.agent.tool_timeout_s
                )
                if output.is_error:
                    tool_span.error("tool_error", str(output.content.get("error", "")))
                tool_span.end()
                state.tool_calls.append(record)
                state.evidence.extend(i for i in output.evidence_ids if i not in state.evidence)
                state.flags.extend(output.flags)
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(output.content, ensure_ascii=False),
                        is_error=output.is_error,
                    )
                )
                yield AgentEvent(type="tool_result", data=record.model_dump(mode="json"))

            if submit is None:
                continue
            if len(turn.tool_calls) > 1:
                # Answer submitted alongside other calls: make the model look at those results first.
                messages.append(
                    self._tool_error(submit, "submit_answer must be called on its own, after reading results")
                )
                continue
            try:
                args = SubmitAnswerArgs.model_validate(submit.arguments)
            except ValidationError as exc:
                messages.append(self._tool_error(submit, f"invalid submit_answer arguments: {exc.errors()}"))
                continue
            return self._finalize(state, args)
        return self._escalate(state, "step_limit_reached")

    def _screen_input(self, state: _RunState) -> str:
        question = state.request.question
        if self.settings.guardrails.redact_input_pii:
            red = pii.redact(question)
            if red.redacted:
                state.flags.append("pii_redacted")
                question = red.text
        if injection.detect(question):
            # Flag and continue: the answer is still constrained to cited, retrieved evidence.
            state.flags.append("injection_suspected_in_question")
        return question

    @staticmethod
    def _tool_error(call: ToolCall, message: str) -> Message:
        return Message(
            role="tool", tool_call_id=call.id, content=json.dumps({"error": message}), is_error=True
        )

    def _finalize(self, state: _RunState, args: SubmitAnswerArgs) -> AgentResponse:
        if args.status == "escalated":
            return self._escalate(state, args.reason or "model_escalated")
        check = grounding.verify(args.answer, args.citations, state.evidence)
        if not check.ok:
            state.flags.append("grounding_failed")
            return self._escalate(state, f"grounding check failed: {check.reason}")
        if check.uncited_sentences:
            state.flags.append("uncited_sentences")
        return self._response(state, "answered", args.answer, check.citations, args.confidence, None)

    def _escalate(self, state: _RunState, reason: str) -> AgentResponse:
        return self._response(state, "escalated", self.settings.agent.escalation_message, [], "low", reason)

    def _response(
        self,
        state: _RunState,
        status: Status,
        answer: str,
        citations: list[str],
        confidence: Confidence,
        reason: str | None,
    ) -> AgentResponse:
        model = state.model or self.llm.model
        price_in, price_out = self.settings.llm.prices.get(model, (0.0, 0.0))
        cost = (state.usage.input_tokens * price_in + state.usage.output_tokens * price_out) / 1_000_000
        return AgentResponse(
            request_id=state.request_id,
            status=status,
            answer=answer,
            citations=citations,
            confidence=confidence,
            reason=reason,
            flags=sorted(set(state.flags)),
            tool_calls=state.tool_calls,
            retrieved_ids=state.evidence,
            usage=state.usage,
            cost_usd=round(cost, 6),
            latency_ms=round((time.perf_counter() - state.started) * 1000, 1),
            steps=state.steps,
            model=model,
            prompt_version=PROMPT_VERSION,
        )


def build_agent(settings: Settings, llm: LLMClient | None = None) -> PolicyAgent:
    from policy_agent.llm import make_llm
    from policy_agent.retrieval.hybrid import get_retriever

    setup_tracing(settings.observability)
    return PolicyAgent(
        settings=settings,
        llm=llm or make_llm(settings.llm),
        retriever=get_retriever(settings.retrieval),
        cases=CaseStore.from_file(settings.retrieval.cases_path),
    )


__all__ = ["AgentEvent", "LLMTurn", "PolicyAgent", "SubmitAnswerArgs", "build_agent"]
