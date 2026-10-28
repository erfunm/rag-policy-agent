"""FastAPI service: ``POST /ask`` (JSON), ``POST /ask/stream`` (SSE), ``GET /healthz``.

The caller's role comes from the ``X-User-Role`` header. In a real deployment an authenticating
gateway sets it from the user's identity; the API never trusts a role named in the question.

Run: ``uvicorn policy_agent.api.app:create_app --factory``
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, get_args

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from policy_agent import __version__
from policy_agent.agent.loop import PolicyAgent, build_agent
from policy_agent.config import get_settings
from policy_agent.models import AgentResponse, AskRequest, Role


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["How long do we keep call recordings?"])
    as_of: date | None = Field(default=None, description="Answer as of this date (default: today).")


RoleHeader = Annotated[str, Header(alias="X-User-Role", description="support | operations | compliance")]


def _role(value: str) -> Role:
    if value not in get_args(Role):
        raise HTTPException(status_code=400, detail=f"X-User-Role must be one of {list(get_args(Role))}")
    return value  # type: ignore[return-value]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(agent: PolicyAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loading the ONNX models takes a few seconds; do it once at startup, not per request.
        app.state.agent = agent or build_agent(get_settings())
        yield

    app = FastAPI(
        title="Policy Agent",
        version=__version__,
        description="Grounded policy Q&A with tool calling, hybrid retrieval and guardrails.",
        lifespan=lifespan,
    )

    def _agent(request: Request) -> PolicyAgent:
        return request.app.state.agent  # type: ignore[no-any-return]

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        a = _agent(request)
        return {
            "status": "ok",
            "version": __version__,
            "provider": a.llm.provider,
            "model": a.llm.model,
            "chunks": len(a.retriever.chunks),
        }

    @app.post("/ask", response_model=AgentResponse)
    def ask(body: AskBody, request: Request, x_user_role: RoleHeader = "support") -> AgentResponse:
        req = AskRequest(question=body.question, as_of=body.as_of, role=_role(x_user_role))
        return _agent(request).ask(req)

    @app.post("/ask/stream", response_class=StreamingResponse)
    def ask_stream(body: AskBody, request: Request, x_user_role: RoleHeader = "support") -> StreamingResponse:
        req = AskRequest(question=body.question, as_of=body.as_of, role=_role(x_user_role))
        agent = _agent(request)

        def events() -> Iterator[str]:
            for event in agent.stream(req):
                yield _sse(event.type, event.data)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app
