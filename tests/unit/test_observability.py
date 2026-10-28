import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from policy_agent.agent.loop import PolicyAgent
from policy_agent.config import ObservabilitySettings
from policy_agent.llm.base import LLMError, LLMTurn, Message, ToolSpec
from policy_agent.llm.fake import FakeLLM
from policy_agent.models import AskRequest
from policy_agent.observability import tracing
from policy_agent.observability.audit import AuditLogger
from policy_agent.observability.tracing import Span, Tracer
from tests.conftest import TODAY


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Tracer(provider)


def _agent(retriever, cases, settings, tracer: Tracer, llm=None) -> PolicyAgent:  # type: ignore[no-untyped-def]
    return PolicyAgent(
        settings=settings, llm=llm or FakeLLM(), retriever=retriever, cases=cases, tracer=tracer
    )


def test_genai_span_tree(retriever, cases, settings, tracer, exporter) -> None:  # type: ignore[no-untyped-def]
    resp = _agent(retriever, cases, settings, tracer).ask(
        AskRequest(question="REQ-9999 status and call recordings retention policy", as_of=TODAY)
    )
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) >= {
        "invoke_agent policy-agent",
        "chat fake-llm",
        "execute_tool search_policies",
        "execute_tool get_case_status",
        "retrieval policy-library",
    }
    root = spans["invoke_agent policy-agent"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["policy_agent.request_id"] == resp.request_id
    assert root.attributes["gen_ai.usage.input_tokens"] == resp.usage.input_tokens
    chat = spans["chat fake-llm"]
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "fake"
    assert chat.parent is not None and chat.parent.span_id == root.context.span_id
    tool = spans["execute_tool get_case_status"]
    assert tool.attributes["gen_ai.tool.name"] == "get_case_status"
    assert tool.attributes["error.type"] == "tool_error"
    retrieval = spans["retrieval policy-library"]
    assert retrieval.parent.span_id == spans["execute_tool search_policies"].context.span_id
    assert retrieval.attributes["gen_ai.data_source.id"] == "policy-library"
    assert all(s.context.trace_id == root.context.trace_id for s in spans.values())


def test_llm_error_marks_chat_span(retriever, cases, settings, tracer, exporter) -> None:  # type: ignore[no-untyped-def]
    class Broken:
        provider = "broken"
        model = "m"

        def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMTurn:
            raise LLMError("boom")

    _agent(retriever, cases, settings, tracer, Broken()).ask(AskRequest(question="hi there"))
    chat = next(s for s in exporter.get_finished_spans() if s.name == "chat m")
    assert chat.attributes["error.type"] == "LLMError"


def test_audit_log_is_redacted_and_correlated(retriever, cases, settings, tracer) -> None:  # type: ignore[no-untyped-def]
    settings.guardrails.redact_input_pii = False  # even then, logs are redacted
    resp = _agent(retriever, cases, settings, tracer).ask(
        AskRequest(question="jo@x.io asks how long call recordings are kept", as_of=TODAY)
    )
    line = json.loads(Path(settings.observability.audit_log_path).read_text().splitlines()[-1])
    assert line["request_id"] == resp.request_id and line["trace_id"] and len(line["trace_id"]) == 32
    assert "jo@x.io" not in json.dumps(line) and "[REDACTED:email]" in line["question"]
    assert line["prompt_version"] == resp.prompt_version and line["status"] == resp.status
    assert line["tool_calls"][0]["name"] == "search_policies"
    assert "[REDACTED:email]" in line["tool_calls"][0]["arguments"]["query"]


def test_audit_logger_appends(tmp_path: Path) -> None:
    log = AuditLogger(tmp_path / "nested" / "a.jsonl")
    log.write({"card": "4111 1111 1111 1111", "n": 1, "items": ["x@y.io"]})
    log.write({"n": 2})
    lines = [json.loads(x) for x in (tmp_path / "nested" / "a.jsonl").read_text().splitlines()]
    assert (
        lines[0] == {"card": "[REDACTED:card]", "n": 1, "items": ["[REDACTED:email]"]} and lines[1]["n"] == 2
    )


def test_noop_span_and_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    span = Span()
    span.set("a", 1)
    span.error("x")
    span.end()
    assert span.trace_id is None
    assert tracing.setup_tracing(ObservabilitySettings(tracing_enabled=False)) is False
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", lambda provider: None)
    assert tracing.setup_tracing(ObservabilitySettings(tracing_enabled=True)) is True
