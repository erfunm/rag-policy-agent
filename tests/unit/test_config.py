from datetime import date

import pytest

from policy_agent.config import Settings
from policy_agent.models import Chunk, Usage


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_AGENT_LLM__PROVIDER", "anthropic")
    monkeypatch.setenv("POLICY_AGENT_RETRIEVAL__TOP_K", "7")
    s = Settings()
    assert s.llm.provider == "anthropic"
    assert s.retrieval.top_k == 7
    assert s.llm.anthropic_model == "claude-sonnet-5-5"


def test_alternate_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from pathlib import Path

    cfg = Path(str(tmp_path)) / "c.yaml"
    cfg.write_text("agent:\n  max_steps: 2\n")
    monkeypatch.setenv("POLICY_AGENT_CONFIG", str(cfg))
    assert Settings().agent.max_steps == 2


def test_chunk_temporal_and_acl() -> None:
    c = Chunk(
        chunk_id="P@1#s1",
        policy_id="P",
        version="1",
        title="t",
        section="s",
        text="x",
        effective_date=date(2024, 1, 1),
        superseded_on=date(2025, 1, 1),
        acl_roles=["compliance"],
    )
    assert c.key == "P@1"
    assert not c.in_force(date(2023, 12, 31))
    assert c.in_force(date(2024, 6, 1))
    assert not c.in_force(date(2025, 1, 1))
    assert c.visible_to("compliance") and not c.visible_to("support")


def test_usage_add() -> None:
    assert Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3) == Usage(
        input_tokens=4, output_tokens=2
    )
