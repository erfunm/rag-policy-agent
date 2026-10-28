import json

import pytest

from policy_agent import cli
from policy_agent.agent import loop
from policy_agent.retrieval import hybrid


@pytest.fixture(autouse=True)
def _fast_backends(monkeypatch: pytest.MonkeyPatch, make_agent, retriever) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(loop, "build_agent", lambda settings, llm=None: make_agent(llm))
    monkeypatch.setattr(hybrid, "build_retriever", lambda settings: retriever)


def test_ask_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", "status of REQ-1001", "--llm", "fake"]) == 0
    out = capsys.readouterr().out
    assert "[ANSWERED]" in out and "CASE:REQ-1001" in out and "tool get_case_status" in out


def test_ask_json_and_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli.main(["ask", "card 4111 1111 1111 1111 dispute deadline", "--json", "--as-of", "2026-10-28"]) == 0
    )
    body = json.loads(capsys.readouterr().out)
    assert "pii_redacted" in body["flags"]


def test_escalated_output(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["ask", "status of REQ-9999"])
    out = capsys.readouterr().out
    assert "[ESCALATED]" in out and "reason=case REQ-9999 not found" in out


def test_ingest(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest"]) == 0
    assert "13 policy versions (12 policies)" in capsys.readouterr().out


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
