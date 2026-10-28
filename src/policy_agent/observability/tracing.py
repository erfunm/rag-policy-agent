"""OpenTelemetry tracing with the GenAI semantic conventions (semantic-conventions-genai, development status).

Spans per request::

    invoke_agent policy-agent     gen_ai.operation.name=invoke_agent (INTERNAL)
    |- chat claude-sonnet-5-5     gen_ai.operation.name=chat, usage + finish reasons (CLIENT)
    |- execute_tool search_policies   gen_ai.operation.name=execute_tool, gen_ai.tool.* (INTERNAL)
    |  `- retrieval policy-library    gen_ai.operation.name=retrieval, gen_ai.data_source.id (CLIENT)

Parents are passed explicitly instead of through the implicit current-span context, because the
agent is a generator that FastAPI may resume on different worker threads while streaming.
Without the ``otel`` extra, or without a configured provider, every call is a cheap no-op.
"""

from __future__ import annotations

from typing import Any, Literal

from policy_agent.config import ObservabilitySettings

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _OTEL = True
except ImportError:  # pragma: no cover - exercised only without the otel extra
    _OTEL = False

Kind = Literal["internal", "client"]
AttrValue = str | bool | int | float | list[str]


class Span:
    """Thin wrapper so callers never touch the OTel API directly (and nothing breaks without it)."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def set(self, key: str, value: AttrValue | None) -> None:
        if self._span is not None and value is not None:
            self._span.set_attribute(key, value)

    def set_many(self, attributes: dict[str, AttrValue | None]) -> None:
        for k, v in attributes.items():
            self.set(k, v)

    def error(self, error_type: str, description: str = "") -> None:
        if self._span is not None:
            self._span.set_attribute("error.type", error_type)
            self._span.set_status(Status(StatusCode.ERROR, description))

    def end(self) -> None:
        if self._span is not None:
            self._span.end()

    @property
    def trace_id(self) -> str | None:
        if self._span is None:
            return None
        ctx = self._span.get_span_context()
        return format(ctx.trace_id, "032x") if ctx.is_valid else None

    @property
    def raw(self) -> Any:
        return self._span


class Tracer:
    def __init__(self, provider: Any = None) -> None:
        self._tracer = None
        if _OTEL:
            provider = provider or trace.get_tracer_provider()
            self._tracer = provider.get_tracer("policy_agent")

    def start(
        self,
        name: str,
        kind: Kind = "internal",
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if self._tracer is None:
            return Span()
        context = (
            trace.set_span_in_context(parent.raw) if parent is not None and parent.raw is not None else None
        )
        span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT if kind == "client" else SpanKind.INTERNAL,
            attributes={k: v for k, v in (attributes or {}).items() if v is not None},
        )
        return Span(span)


_configured = False


def setup_tracing(settings: ObservabilitySettings) -> bool:
    """Install an OTLP/HTTP exporter (endpoint from OTEL_EXPORTER_OTLP_ENDPOINT). Idempotent."""
    global _configured
    if _configured or not settings.tracing_enabled or not _OTEL:
        return _configured
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    return True
