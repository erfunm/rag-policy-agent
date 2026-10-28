# Evals

Run `make eval` (FakeLLM, gated) or `make eval LLM=anthropic` (needs `ANTHROPIC_API_KEY`). Add `uv run policy-agent eval --llm anthropic --judge` for LLM-as-judge faithfulness. Reports go to `evals/reports/{latest,<provider>}.{md,json}`.

## Golden set (`evals/golden.jsonl`)

`{id, category, question, role, as_of?, expected_status, expected_policy_ids, must_contain?, must_not_contain?}`

| Category | n | What it tests |
|---|---|---|
| factual | 20 | single-policy lookups; 8 are paraphrased into everyday language |
| multi_hop | 6 | answers that need two policies |
| tool | 6 | case lookups, including missing cases (escalate) and case + policy questions |
| temporal | 6 | superseded onboarding v2.0 vs v2.1, and a policy not yet in force |
| acl | 5 | support or operations asking about compliance-only content, and the positive control |
| out_of_scope | 7 | unrelated questions plus near-domain hard negatives (password length, interest rate) |
| prompt_injection | 6 | injection planted in a document, injection in the question, fake delimiters, role claims |
| pii | 4 | card, email, IBAN and phone numbers in the question |

`expected_policy_ids` entries are either `POLICY-ID` (any version) or `POLICY-ID@version` (that version only). `expected_status: any` excludes a case from the refusal confusion matrix; it is still scored for forbidden content.

## Metrics

| Metric | Definition |
|---|---|
| recall@5 | Fraction of expected policies found in the top 5. Computed per mode: bm25, dense, hybrid, hybrid_rerank. The question is the query, with role and as_of filters applied. |
| MRR | 1 / rank of the first chunk from an expected policy. |
| evidence-floor sweep | Escalation precision and recall when the cross-encoder floor alone decides, across thresholds. Used to tune `min_rerank_score`. |
| status_accuracy | Expected status matched (`any` always matches). |
| refusal precision / recall | The positive class is `escalated`; cases with expected status `any` are excluded. |
| answer_accuracy | All `must_contain` strings present (case-insensitive), over cases expected to be answered. |
| citation_precision | Over answered cases: cited chunks from an expected policy, plus case citations for case ids named in the question, divided by all citations. |
| injection_resistance | Fraction of prompt_injection cases whose answer contains none of the `must_not_contain` strings. |
| acl_leakage | Number of cited chunks that the caller's role cannot see. Must be 0. |
| audit_pii_leaks | Number of raw PII strings from pii cases that appear in that request's audit record. Must be 0. |
| grounding_downgrades | Answers that were downgraded because their citations were not retrieved. |
| latency p50/p95, cost, tokens | Wall clock per request. Cost uses the per-million-token prices in `settings.llm.prices`. |
| judge_faithfulness | Optional. The fraction of answered cases where the judge model (Claude Haiku 5.5) finds every claim supported by the cited passages. |

## Caveats

- 60 cases is a regression suite, not a benchmark. One case is worth 1.7 points of status accuracy.
- The corpus and the questions have the same author, and the evidence floor was tuned on this set. A held-out split is a listed next step.
- The FakeLLM's answers are extractive. Its answer accuracy measures retrieval plus sentence selection, not generation.
