import argparse
import json
from pathlib import Path

import pytest

from policy_agent import evaluation
from policy_agent.evaluation import GoldenCase, apply_gate, load_golden, render_markdown, run_eval

GOLDEN = Path("evals/golden.jsonl")


def test_golden_set_is_valid_and_balanced() -> None:
    cases = load_golden(GOLDEN)
    assert 40 <= len(cases) <= 60
    assert len({c.id for c in cases}) == len(cases)
    cats = {c.category for c in cases}
    assert cats == {
        "factual",
        "multi_hop",
        "tool",
        "temporal",
        "acl",
        "out_of_scope",
        "prompt_injection",
        "pii",
    }


def _subset() -> list[GoldenCase]:
    wanted = {
        "factual-01",
        "temporal-01",
        "tool-04",
        "acl-01",
        "prompt_injection-02",
        "out_of_scope-03",
        "pii-02",
    }
    return [c for c in load_golden(GOLDEN) if c.id in wanted]


def test_run_eval_scores_and_gates(make_agent, settings) -> None:  # type: ignore[no-untyped-def]
    thresholds = {"fake": {"agent.acl_leakage": {"max": 0}, "agent.status_accuracy": {"min": 1.01}}}
    report = run_eval(settings, _subset(), thresholds, agent=make_agent())
    assert report["meta"]["cases"] == 7
    assert {r["mode"] for r in report["retrieval"]} == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert report["agent"]["acl_leakage"] == 0
    assert report["agent"]["injection_resistance"] == 1.0
    assert report["evidence_floor_sweep"][0]["cases"] > 0
    assert not report["gate"]["passed"]  # impossible min on status_accuracy
    assert [c["passed"] for c in report["gate"]["checks"]] == [True, False]
    md = render_markdown(report)
    assert "## Retrieval ablation" in md and "FAILED" in md and "hybrid_rerank" in md


def test_gate_bounds_and_missing_metrics() -> None:
    report = {"agent": {"a": 0.5, "b": 3}}
    gate = apply_gate(
        report, {"default": {"agent.a": {"min": 0.4}, "agent.b": {"max": 2}, "agent.c": {"min": 0}}}, "x"
    )
    assert [c["passed"] for c in gate.checks] == [True, False, False] and not gate.passed
    assert apply_gate(report, {}, "x").passed


def test_scoring_of_citations_and_leaks(make_agent, retriever) -> None:  # type: ignore[no-untyped-def]
    from policy_agent.models import AgentResponse

    case = GoldenCase(
        id="x",
        category="acl",
        question="REQ-1001 SAR",
        expected_status="answered",
        expected_policy_ids=["PRC-SAR-001"],
    )
    resp = AgentResponse(
        request_id="r",
        status="answered",
        answer="text",
        citations=["PRC-SAR-001@1.0#s2", "CASE:REQ-1001", "POL-SAN-004@1.3#s1"],
        confidence="high",
        model="m",
        prompt_version="p",
    )
    scored = evaluation.score_case(case, resp, retriever)
    assert scored.acl_leaks == ["PRC-SAR-001@1.0#s2"]
    assert scored.correct_citations == 2 and scored.total_citations == 3


def test_main_writes_reports(
    make_agent,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(evaluation, "build_agent", lambda settings, llm=None: make_agent())
    golden = tmp_path / "g.jsonl"
    golden.write_text("\n".join(c.model_dump_json() for c in _subset()))
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("fake:\n  agent.acl_leakage: {max: 0}\n")
    args = argparse.Namespace(
        golden=str(golden),
        thresholds=str(thresholds),
        out=str(tmp_path / "out"),
        llm=None,
        mode="hybrid",
        judge=False,
        no_gate=False,
        limit=None,
    )
    assert evaluation.main(args) == 0
    data = json.loads((tmp_path / "out" / "latest.json").read_text())
    assert data["gate"]["passed"] and (tmp_path / "out" / "fake.md").exists()
    thresholds.write_text("fake:\n  agent.status_accuracy: {min: 2}\n")
    assert evaluation.main(args) == 1
    args.no_gate = True
    assert evaluation.main(args) == 0
