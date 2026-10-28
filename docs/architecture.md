# Architecture

## Request lifecycle

1. **Entry.** The CLI, `POST /ask`, `POST /ask/stream` or the Streamlit UI builds an `AskRequest` (`question`, `role`, `as_of`). The role comes from the caller's identity (`X-User-Role`, set by a gateway in production), never from the question text.
2. **Input guardrails** (`guardrails/pii.py`, `guardrails/injection.py`). PII is redacted to `[REDACTED:type]` before the LLM sees the question; this is configurable. Injection heuristics flag the request but do not block it, because the answer is still constrained to cited evidence.
3. **Agent loop** (`agent/loop.py`):
   - send the system prompt, the transcript and the tool specs to the `LLMClient`;
   - for each tool call: validate the arguments with pydantic, run with a timeout, and append the result (errors included) as a tool message;
   - repeat until the model calls `submit_answer` or `max_steps` is reached;
   - a plain-text reply gets one nudge, then escalates;
   - a `submit_answer` sent alongside other calls is deferred until the model has seen those results;
   - an invalid `submit_answer` is returned to the model as an error.
4. **Tools** (`tools/`). The role and date in `ToolContext` come from the request.
   - `search_policies`: hybrid retrieval. Results carry `chunk_id`, `evidence_sufficient` and `access_restricted`.
   - `get_policy`: the version in force on `as_of`, with all its sections.
   - `get_case_status`: looks the case up in `data/cases.json`.
5. **Retrieval** (`retrieval/hybrid.py`):
   - Candidates = chunks that are in force on `as_of` and visible to the role.
   - Each method ranks the candidates: BM25 (`rank-bm25`) and dense cosine (fastembed `bge-small-en-v1.5`, embeddings cached in `.cache/index`).
   - The two rankings are fused with RRF (k = 60) over the top 20 of each.
   - The top 20 fused results are reranked by the `ms-marco-MiniLM-L-6-v2` cross-encoder, and the top 5 are returned.
6. **Spotlighting.** Retrieved text is sanitised (sentences that look like instructions are replaced by a marker) and wrapped in `<untrusted_document id=...>`.
7. **Grounding** (`guardrails/grounding.py`). Declared and inline `[id]` citations must be a subset of the evidence ids that tools returned in this request. Otherwise the answer becomes `escalated` with `grounding_failed`.
8. **Telemetry.** One OTel trace per request and one PII-redacted JSONL audit record, linked by `trace_id`.

## LLM abstraction

```python
class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn: ...
```

`Message` and `LLMTurn` are provider-neutral. Each adapter translates them to and from its wire format.

- **Claude** (`llm/anthropic.py`):
  - Uses the Messages API with native `tool_use` / `tool_result` blocks.
  - All results for one turn go back in one user message.
  - Assistant content is replayed verbatim (`Message.raw`), which keeps thinking blocks valid.
  - The system prompt and tools are cached with `cache_control`, and effort is configurable.
  - Server-side refusal fallback is on (`fallbacks="default"`), and `stop_reason="refusal"` becomes an escalation.
- **OpenAI-compatible** (`llm/openai_compat.py`): `chat.completions` with function tools. `base_url` covers Ollama, vLLM and similar servers. Unparseable tool arguments go back to the model as validation errors.
- **FakeLLM** (`llm/fake.py`):
  - Plans tool calls from the question (case ids → `get_case_status`, policy intent → `search_policies`).
  - Reads tool results from the transcript.
  - Writes an extractive cited answer, or escalates on errors, restricted hits or a low evidence floor.

## Data model

Policies are markdown files with YAML front-matter (`policy_id`, `version`, `effective_date`, `owner`, `acl_roles`, `summary`). The loader groups versions by `policy_id` and sets `superseded_on` from the next version's `effective_date`. Chunk ids are `{policy_id}@{version}#s{section}[.{part}]`, so they only change when the document structure changes.

## Deployment

The `Dockerfile` has two stages:
1. **Build:** uv installs the locked dependencies (api + otel extras only), then `policy-agent ingest` downloads the models and builds the embedding index into the image.
2. **Runtime:** a slim Python image running as a non-root user, with a healthcheck on `/healthz`.

`docker-compose.yml` adds Jaeger 2.x. It receives OTLP over HTTP on port 4318, and the UI is on port 16686.
