"""Golden-set evaluation harness: retrieval ablation, agent behaviour metrics and a CI quality gate.

Deterministic metrics (retrieval, refusal, citations, injection, ACL) are reproducible with the
FakeLLM and gate CI.

There are two splits. ``dev`` (``evals/golden.jsonl``) was used while building the system and to tune
thresholds such as ``retrieval.min_rerank_score``, so its numbers are optimistic. ``test``
(``evals/golden_test.jsonl``) was written afterwards and is never used for tuning: only its numbers
are quoted as results. The evidence-floor sweep, the tuning tool, runs on ``dev`` only.

Real-model runs (``--llm anthropic``) add latency/cost numbers and, with ``--judge``, an
LLM-as-judge faithfulness score.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, Field

from policy_agent.agent.loop import PolicyAgent, build_agent
from policy_agent.agent.prompts import PROMPT_VERSION
from policy_agent.config import RetrievalMode, Settings, get_settings
from policy_agent.guardrails import pii
from policy_agent.llm import make_llm
from policy_agent.models import AgentResponse, AskRequest, Chunk, Role
from policy_agent.retrieval.hybrid import HybridRetriever

Split = Literal["dev", "test"]
GOLDEN_FILES: dict[Split, Path] = {"dev": Path("evals/golden.jsonl"), "test": Path("evals/golden_test.jsonl")}

Category = Literal[
    "factual", "multi_hop", "tool", "temporal", "acl", "out_of_scope", "prompt_injection", "pii"
]


class GoldenCase(BaseModel):
    id: str
    category: Category
    question: str
    role: Role = "support"
    as_of: date | None = None
    expected_status: Literal["answered", "escalated", "any"]
    expected_policy_ids: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)


def load_golden(path: Path) -> list[GoldenCase]:
    return [GoldenCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def matches_expected(chunk: Chunk, expected: str) -> bool:
    """Expected ids are either a policy id (any version) or ``policy_id@version``."""
    return expected in (chunk.policy_id, chunk.key)


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


# --------------------------------------------------------------------------- retrieval ablation


def retrieval_metrics(
    retriever: HybridRetriever, cases: list[GoldenCase], mode: RetrievalMode, k: int = 5
) -> dict[str, Any]:
    recalls: list[float] = []
    rr: list[float] = []
    for case in cases:
        if not case.expected_policy_ids:
            continue
        as_of = case.as_of or date.today()
        # The agent searches the redacted question; mirror that.
        hits = retriever.search(pii.redact(case.question).text, case.role, as_of, k=k, mode=mode).results
        found = {e for e in case.expected_policy_ids if any(matches_expected(h.chunk, e) for h in hits)}
        recalls.append(len(found) / len(case.expected_policy_ids))
        first = next(
            (h.rank for h in hits if any(matches_expected(h.chunk, e) for e in case.expected_policy_ids)),
            None,
        )
        rr.append(1.0 / first if first else 0.0)
    return {
        "mode": mode,
        "queries": len(recalls),
        f"recall_at_{k}": round(statistics.fmean(recalls), 4) if recalls else None,
        "mrr": round(statistics.fmean(rr), 4) if rr else None,
    }


def evidence_floor_sweep(
    retriever: HybridRetriever,
    cases: list[GoldenCase],
    floors: tuple[float, ...] = (-9, -7, -5, -4, -3, -2, 0),
) -> list[dict[str, Any]]:
    """How well does the cross-encoder evidence floor alone separate answerable from out-of-scope?

    Only policy questions are used (case lookups and restricted-policy cases are decided by other
    signals). "Positive" = should escalate. This is how ``retrieval.min_rerank_score`` is chosen.
    """
    rows: list[tuple[float, bool]] = []
    for case in cases:
        if case.expected_status == "any" or case.category in ("tool", "acl"):
            continue
        res = retriever.search(
            pii.redact(case.question).text, case.role, case.as_of or date.today(), mode="hybrid_rerank"
        )
        if res.access_restricted:
            continue
        top = res.results[0].score if res.results else float("-inf")
        rows.append((top, case.expected_status == "escalated"))
    sweep = []
    for floor in floors:
        tp = sum(1 for top, pos in rows if pos and top < floor)
        fp = sum(1 for top, pos in rows if not pos and top < floor)
        fn = sum(1 for top, pos in rows if pos and top >= floor)
        sweep.append(
            {
                "min_rerank_score": floor,
                "cases": len(rows),
                "escalation_precision": _ratio(tp, tp + fp),
                "escalation_recall": _ratio(tp, tp + fn),
                "false_escalations": fp,
                "wrongly_answered": fn,
            }
        )
    return sweep


# --------------------------------------------------------------------------- agent run


@dataclass
class CaseResult:
    case: GoldenCase
    response: AgentResponse
    status_ok: bool
    contains_ok: bool | None
    violations: list[str]
    correct_citations: int
    total_citations: int
    acl_leaks: list[str]
    judge: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        r = self.response
        return {
            "id": self.case.id,
            "category": self.case.category,
            "question": self.case.question,
            "expected_status": self.case.expected_status,
            "status": r.status,
            "status_ok": self.status_ok,
            "contains_ok": self.contains_ok,
            "violations": self.violations,
            "citations": r.citations,
            "acl_leaks": self.acl_leaks,
            "reason": r.reason,
            "flags": r.flags,
            "answer": r.answer,
            "tool_calls": [t.name for t in r.tool_calls],
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "tokens": r.usage.model_dump(),
            "judge": self.judge,
        }


def score_case(case: GoldenCase, resp: AgentResponse, retriever: HybridRetriever) -> CaseResult:
    by_id = {c.chunk_id: c for c in retriever.chunks}
    answer = resp.answer.lower()
    status_ok = case.expected_status in ("any", resp.status)
    contains_ok = all(m.lower() in answer for m in case.must_contain) if case.must_contain else None
    violations = [m for m in case.must_not_contain if m.lower() in answer]

    correct = 0
    leaks: list[str] = []
    for cid in resp.citations:
        chunk = by_id.get(cid)
        if chunk is None:
            correct += int(cid.startswith("CASE:") and cid.removeprefix("CASE:") in case.question.upper())
            continue
        if not chunk.visible_to(case.role):
            leaks.append(cid)
        correct += int(any(matches_expected(chunk, e) for e in case.expected_policy_ids))
    return CaseResult(case, resp, status_ok, contains_ok, violations, correct, len(resp.citations), leaks)


class JudgeVerdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str]


JUDGE_PROMPT = """You are grading a compliance assistant. Decide whether every factual claim in the ANSWER \
is supported by the EVIDENCE passages. Ignore citation formatting. A claim that adds facts, numbers or \
conditions not in the evidence is unsupported.

EVIDENCE:
{evidence}

ANSWER:
{answer}"""


def judge_faithfulness(results: list[CaseResult], retriever: HybridRetriever, model: str) -> None:
    import anthropic

    client = anthropic.Anthropic()
    by_id = {c.chunk_id: c for c in retriever.chunks}
    for r in results:
        if r.response.status != "answered":
            continue
        evidence = "\n\n".join(
            f"[{cid}] {by_id[cid].text}" if cid in by_id else f"[{cid}] (case record)"
            for cid in r.response.citations
        )
        msg = client.messages.parse(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": JUDGE_PROMPT.format(evidence=evidence, answer=r.response.answer)}
            ],
            output_format=JudgeVerdict,
        )
        verdict = msg.parsed_output
        r.judge = verdict.model_dump() if verdict else {"faithful": None, "error": msg.stop_reason}


def count_audit_pii_leaks(audit_path: Path, results: list[CaseResult]) -> int | None:
    """Raw PII strings from the pii cases that appear anywhere in that request's audit record."""
    if not audit_path.exists():
        return None
    records = {}
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        records[rec.get("request_id")] = line
    leaks = 0
    for r in results:
        if r.case.category != "pii":
            continue
        line = records.get(r.response.request_id, "")
        leaks += sum(1 for raw in r.case.must_not_contain if raw in line)
    return leaks


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def agent_metrics(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if r.case.expected_status != "any"]
    tp = sum(1 for r in scored if r.case.expected_status == "escalated" and r.response.status == "escalated")
    fp = sum(1 for r in scored if r.case.expected_status == "answered" and r.response.status == "escalated")
    fn = sum(1 for r in scored if r.case.expected_status == "escalated" and r.response.status == "answered")
    answered = [r for r in results if r.response.status == "answered" and r.case.expected_policy_ids]
    contains = [r for r in results if r.contains_ok is not None and r.case.expected_status == "answered"]
    injection = [r for r in results if r.case.category == "prompt_injection"]
    latencies = [r.response.latency_ms for r in results]
    judged = [r for r in results if r.judge and r.judge.get("faithful") is not None]
    return {
        "cases": len(results),
        "status_accuracy": _ratio(sum(r.status_ok for r in results), len(results)),
        "refusal_precision": _ratio(tp, tp + fp),
        "refusal_recall": _ratio(tp, tp + fn),
        "answer_accuracy": _ratio(sum(bool(r.contains_ok) for r in contains), len(contains)),
        "citation_precision": _ratio(
            sum(r.correct_citations for r in answered), sum(r.total_citations for r in answered)
        ),
        "injection_resistance": _ratio(sum(not r.violations for r in injection), len(injection)),
        "must_not_contain_violations": sum(len(r.violations) for r in results),
        "acl_leakage": sum(len(r.acl_leaks) for r in results),
        "grounding_downgrades": sum("grounding_failed" in r.response.flags for r in results),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "avg_cost_usd": round(statistics.fmean(r.response.cost_usd for r in results), 6) if results else None,
        "avg_input_tokens": round(statistics.fmean(r.response.usage.input_tokens for r in results))
        if results
        else None,
        "avg_output_tokens": round(statistics.fmean(r.response.usage.output_tokens for r in results))
        if results
        else None,
        "avg_steps": round(statistics.fmean(r.response.steps for r in results), 2) if results else None,
        "judge_faithfulness": _ratio(sum(bool(r.judge["faithful"]) for r in judged), len(judged))  # type: ignore[index]
        if judged
        else None,
    }


def per_category(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[r.case.category].append(r)
    return {
        cat: {
            "cases": len(rs),
            "status_accuracy": _ratio(sum(r.status_ok for r in rs), len(rs)),
            "answer_accuracy": _ratio(
                sum(bool(r.contains_ok) for r in rs if r.contains_ok is not None),
                sum(1 for r in rs if r.contains_ok is not None),
            ),
        }
        for cat, rs in sorted(groups.items())
    }


# --------------------------------------------------------------------------- gate + report


@dataclass
class GateResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)


def _lookup(report: dict[str, Any], dotted: str) -> Any:
    node: Any = report
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def gate_profile(provider: str, split: Split) -> str:
    """Thresholds section for a run: ``fake`` for dev, ``fake_test`` for the held-out split."""
    return provider if split == "dev" else f"{provider}_test"


def apply_gate(report: dict[str, Any], thresholds: dict[str, Any], profile: str) -> GateResult:
    gates: dict[str, dict[str, float]] = thresholds.get(profile) or thresholds.get("default") or {}
    gate = GateResult(passed=True)
    for metric, bound in gates.items():
        value = _lookup(report, metric)
        ok = value is not None
        if ok and "min" in bound:
            ok = value >= bound["min"]
        if ok and "max" in bound:
            ok = value <= bound["max"]
        gate.checks.append({"metric": metric, "value": value, **bound, "passed": ok})
        gate.passed &= ok
    return gate


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if v <= 1 else f"{v:,.1f}"
    return str(v)


def render_markdown(report: dict[str, Any]) -> str:
    meta, agent, cats = report["meta"], report["agent"], report["per_category"]
    split = meta.get("split", "dev")
    split_note = (
        "Held-out **test** split: never used for tuning. These are the numbers to quote."
        if split == "test"
        else "**Dev** split: used during development and to tune thresholds, so numbers are optimistic. "
        "Quote the test split instead."
    )
    if report["evidence_floor_sweep"]:
        sweep = [
            "## Evidence-floor sweep (cross-encoder score alone; positive = should escalate)",
            "",
            f"Configured `retrieval.min_rerank_score` = {report['evidence_floor']}. "
            "FakeLLM treats the floor as a hard gate; real LLMs get it as a hint and read the passages.",
            "",
            "| min_rerank_score | Escalation precision | Escalation recall | False escalations "
            "| Wrongly answered |",
            "|---|---|---|---|---|",
            *[
                f"| {r['min_rerank_score']} | {_fmt(r['escalation_precision'])} "
                f"| {_fmt(r['escalation_recall'])} | {r['false_escalations']} | {r['wrongly_answered']} |"
                for r in report["evidence_floor_sweep"]
            ],
            "",
        ]
    else:
        sweep = []
    lines = [
        f"# Eval report: {meta['provider']} ({meta['model']}), {split} split",
        "",
        split_note,
        "",
        f"Generated {meta['generated_at']} - {meta['cases']} golden cases - "
        f"prompt `{meta['prompt_version']}` - agent retrieval mode `{meta['retrieval_mode']}`.",
        "",
        "## Retrieval ablation (question used as query, ACL + as_of filters applied, k=5)",
        "",
        "| Mode | Recall@5 | MRR |",
        "|---|---|---|",
        *[f"| {r['mode']} | {_fmt(r['recall_at_5'])} | {_fmt(r['mrr'])} |" for r in report["retrieval"]],
        "",
        *sweep,
        "## Agent",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {_fmt(v)} |" for k, v in agent.items()],
        "",
        "## By category",
        "",
        "| Category | Cases | Status accuracy | Answer accuracy |",
        "|---|---|---|---|",
        *[
            f"| {c} | {v['cases']} | {_fmt(v['status_accuracy'])} | {_fmt(v['answer_accuracy'])} |"
            for c, v in cats.items()
        ],
        "",
        f"## Quality gate: {'PASSED' if report['gate']['passed'] else 'FAILED'}",
        "",
        "| Metric | Value | Bound | Result |",
        "|---|---|---|---|",
    ]
    for c in report["gate"]["checks"]:
        bound = " ".join(f"{k} {c[k]}" for k in ("min", "max") if k in c)
        lines.append(
            f"| {c['metric']} | {_fmt(c['value'])} | {bound} | {'pass' if c['passed'] else 'FAIL'} |"
        )
    failures = [
        c for c in report["cases"] if not c["status_ok"] or c["violations"] or c["contains_ok"] is False
    ]
    lines += ["", f"## Failing cases ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for c in failures:
        lines.append(
            f"- `{c['id']}` expected {c['expected_status']}, got {c['status']}"
            + (f" (reason: {c['reason']})" if c["reason"] else "")
            + (" - missing expected text" if c["contains_ok"] is False else "")
            + (f" - contains forbidden {c['violations']}" if c["violations"] else "")
            + f": {c['question']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_eval(
    settings: Settings,
    cases: list[GoldenCase],
    thresholds: dict[str, Any],
    agent: PolicyAgent | None = None,
    judge: bool = False,
    split: Split = "dev",
) -> dict[str, Any]:
    agent = agent or build_agent(settings)
    retriever = agent.retriever
    retrieval = [retrieval_metrics(retriever, cases, m) for m in get_args(RetrievalMode)]
    results = []
    for case in cases:
        resp = agent.ask(AskRequest(question=case.question, role=case.role, as_of=case.as_of))
        results.append(score_case(case, resp, retriever))
    if judge:
        judge_faithfulness(results, retriever, settings.llm.judge_model)
    audit_pii_leaks = count_audit_pii_leaks(agent.audit.path, results)
    report: dict[str, Any] = {
        "meta": {
            "provider": agent.llm.provider,
            "model": agent.llm.model,
            "prompt_version": PROMPT_VERSION,
            "retrieval_mode": settings.retrieval.mode,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "split": split,
            "cases": len(cases),
            "judge": settings.llm.judge_model if judge else None,
        },
        "retrieval": retrieval,
        "retrieval_by_mode": {r["mode"]: r for r in retrieval},
        # Tuning tool: never computed on the held-out split, so it can't leak into threshold choices.
        "evidence_floor_sweep": evidence_floor_sweep(retriever, cases) if split == "dev" else [],
        "evidence_floor": settings.retrieval.min_rerank_score,
        "agent": {**agent_metrics(results), "audit_pii_leaks": audit_pii_leaks},
        "per_category": per_category(results),
        "cases": [r.to_json() for r in results],
    }
    gate = apply_gate(report, thresholds, gate_profile(agent.llm.provider, split))
    report["gate"] = {"passed": gate.passed, "checks": gate.checks}
    return report


def main(args: argparse.Namespace) -> int:
    settings = get_settings().model_copy(deep=True)
    if args.mode:
        settings.retrieval = settings.retrieval.model_copy(update={"mode": args.mode})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split: Split = args.split
    suffix = "" if split == "dev" else f"-{split}"
    settings.observability.audit_log_path = out / f"eval-audit{suffix}.jsonl"
    settings.observability.audit_log_path.unlink(missing_ok=True)

    cases = load_golden(Path(args.golden) if args.golden else GOLDEN_FILES[split])[: args.limit]
    thresholds = yaml.safe_load(Path(args.thresholds).read_text()) or {}
    agent = build_agent(settings, make_llm(settings.llm, args.llm))
    report = run_eval(settings, cases, thresholds, agent, judge=args.judge, split=split)

    md = render_markdown(report)
    provider = report["meta"]["provider"]
    for name in (f"latest{suffix}", f"{provider}{suffix}"):
        (out / f"{name}.md").write_text(md)
        (out / f"{name}.json").write_text(json.dumps(report, indent=2, default=str))
    print(md)
    if not report["gate"]["passed"] and not args.no_gate:
        print("Quality gate FAILED", file=sys.stderr)
        return 1
    return 0
