UV ?= uv
LLM ?= fake

.PHONY: install lint format type test eval demo up down ingest mcp ui

install:  ## install the package with dev + all runtime extras
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

type:
	$(UV) run mypy

test:
	$(UV) run pytest --cov --cov-report=term --cov-fail-under=85

ingest:  ## build and cache the BM25 + dense index (downloads the ONNX models once)
	$(UV) run policy-agent ingest

eval:  ## run the golden-set eval; LLM=anthropic|openai for real-model numbers
	$(UV) run policy-agent eval --llm $(LLM)

demo:  ## offline demo with FakeLLM, no API key needed
	$(UV) run policy-agent ask "What proof of address is required for a customer sending \$$6,000 a month?"
	$(UV) run policy-agent ask "What proof of address is required for a customer sending \$$6,000 a month?" --as-of 2024-01-01
	$(UV) run policy-agent ask "What is the status of case REQ-9999?"

up:
	docker compose up --build -d

down:
	docker compose down

mcp:
	$(UV) run policy-agent-mcp

ui:
	$(UV) run --extra ui streamlit run ui/app.py
