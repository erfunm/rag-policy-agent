# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-editable --extra api --extra otel --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --extra api --extra otel
COPY configs ./configs
COPY data ./data
# Bake the ONNX models and the embedding index into the image so containers start fast and offline.
RUN .venv/bin/policy-agent ingest

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv ./.venv
COPY --from=build --chown=app:app /app/.cache ./.cache
COPY --from=build --chown=app:app /app/configs ./configs
COPY --from=build --chown=app:app /app/data ./data
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    POLICY_AGENT_OBSERVABILITY__AUDIT_LOG_PATH=/app/logs/audit.jsonl
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "policy_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
