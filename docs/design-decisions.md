# Design decisions (ADR log)

## ADR-1: Hand-written agent loop, no LangChain or LlamaIndex
**Context.** The core value of this project lies in how the agent behaves at the edges: tool errors, step limits, invalid structured output, ungrounded citations, refusals.
**Decision.** Write the plan → act → observe loop by hand behind a small `LLMClient` protocol.
**Consequences.**
- About 340 lines, with every branch unit-tested via `ScriptedLLM`.
- No framework upgrade churn.
- Adding a provider means writing an adapter of about 100 lines.
- We do not get framework integrations for free.

## ADR-2: Structured final answer via a `submit_answer` tool
**Decision.** The model must finish by calling `submit_answer(status, answer, citations, confidence, reason)`, validated with pydantic.
**Why not provider JSON mode?** A terminal tool works the same way across Claude, OpenAI-compatible servers and the FakeLLM. Current Claude models also reject forced `tool_choice`, so the prompt asks for the call and the loop enforces it: one nudge, then escalate.

## ADR-3: Hybrid retrieval, RRF, cross-encoder rerank, all on CPU
**Decision.**
- BM25 handles exact terms and amounts; `bge-small-en-v1.5` handles paraphrases.
- RRF (k = 60) fuses the two rankings without calibrating their scores.
- The `ms-marco-MiniLM-L-6-v2` cross-encoder reranks the top 20.
- Everything runs on ONNX through fastembed, with no GPU and no vector DB.

**Evidence.** On the held-out test split, every mode reaches recall@5 = 1.0, and MRR goes 0.958 (BM25 or dense alone) → 0.972 (RRF) → 0.979 (RRF + rerank). On the dev split, the reranker moves MRR from 0.92 to 1.00 (partly fitted, since dev was used for tuning).

**Why plain RRF can lose to dense-only (dev: MRR 0.919 vs 0.949).**
- RRF weights both rankings equally and ignores how confident each one is. When BM25 is confidently *wrong*, it still pulls its top hit up the fused list.
- That happens with slang paraphrases (*"how long do we hang on to the audio?"*). They share few words with the right policy, but common words like "customer", "days" and "account" appear in almost all 12 policies.
- With only about 50 chunks, BM25's IDF weights are noisy: a word that appears in 3 of 12 documents already looks "rare".
- Dense retrieval handles exactly these questions well, so equal-weight fusion dilutes it.

On a bigger corpus, or with questions that use exact identifiers (`REQ-1001`, `$5,000`, `SAR`), BM25 earns its weight back. That is why the test split, written closer to policy wording, shows RRF above both single modes. Rather than tune RRF weights on 60 questions, the cross-encoder makes the final call: it reads the query and passage together and corrects the fused ordering. RRF's job is just to get the right chunk into the top 20 candidates, and it does that on both splits.

## ADR-4: Filter by ACL and effective date before ranking
**Decision.** Restricted chunks and chunks not in force on `as_of` never enter the candidate set. Separately, the search reports `access_restricted=true` when the best match over all roles is hidden from the caller. That reveals the existence of a restricted match, never its content.
**Trade-off.** Telling the user "this is covered by a policy you cannot see, escalate to Compliance" is more useful than a confident answer drawn from the next-best visible document. It does leak existence. This is accepted, and documented.

## ADR-5: Evidence floor tuned on a sweep, and used differently by FakeLLM and real models
**Context.** The sweep in the eval report shows that no cross-encoder threshold separates paraphrased in-scope questions from near-domain out-of-scope ones.
**Decision.**
- `min_rerank_score = -3.0`, the knee of escalation recall (0.889, with 1 question wrongly answered).
- FakeLLM treats the floor as a hard gate.
- Real models receive `evidence_sufficient` as a hint and are told to answer only if a passage clearly states the answer.

**Why.** In compliance, a wrong answer costs more than an escalation.

## ADR-6: Deterministic FakeLLM for tests, CI and the demo
**Decision.** A rule-based LLM that speaks the same tool-calling protocol.
**Consequences.**
- CI needs no secrets and costs nothing.
- The retrieval, refusal, ACL, injection and PII metrics are exactly reproducible and can gate merges.
- FakeLLM numbers say nothing about generation quality. That is measured with `make eval LLM=anthropic`, gated only on safety properties.

## ADR-7: Prompt-injection defence in depth
**Decision.**
1. Spotlight retrieved text in `<untrusted_document>` delimiters that documents cannot close.
2. Strip sentences that match injection heuristics (configurable, so the model's own resistance can be measured).
3. Keep tools read-only.
4. Verify every citation against evidence retrieved in the request.

Heuristics are a tripwire; steps 3 and 4 are the structural backstop.

## ADR-8: Thresholds and the CI gate
**Decision.** `evals/thresholds.yaml` has separate bounds per provider and split.
- Thresholds such as `min_rerank_score` are tuned on the **dev** split only. The held-out **test** split (`evals/golden_test.jsonl`) was written after development, and its first run is the reported result. The evidence-floor sweep is not computed on it.
- The test split is gated only on safety properties (`fake_test`). Its quality numbers are reported, never tuned against. If a test case starts failing, the fix goes in the system, never in the test case.
- FakeLLM bounds sit just below the measured values and catch regressions.
- Real-model bounds cover safety properties that must hold on every run: injection resistance 1.0, ACL leakage 0, raw PII in the audit log 0, escalation recall ≥ 0.9, citation precision ≥ 0.9.

Raise a bound when an improvement lands. Lower one only together with a written reason in the PR.

## ADR-9: RAGAS not adopted (October 2026)
**Context.** RAGAS remains the common vocabulary for faithfulness and context precision. Its 0.4 line reorganised the metrics API, and it depends on LangChain.
**Decision.** Keep in-house metrics: recall@k and MRR at policy level, citation precision, refusal precision and recall, injection resistance, ACL and PII leakage. Faithfulness gets an optional LLM-as-judge (`--judge`, Claude Haiku 5.5, structured output).
**Revisit when** the corpus or the team grows enough that standard metrics matter more than the dependency weight.
