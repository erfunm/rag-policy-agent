import time
from datetime import date

from pydantic import BaseModel

from policy_agent.llm.base import ToolCall
from policy_agent.tools import default_registry
from policy_agent.tools.registry import Tool, ToolContext, ToolOutput, ToolRegistry


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="t1", name=name, arguments=dict(arguments))


def test_specs_are_json_schema() -> None:
    specs = {s.name: s for s in default_registry().specs()}
    assert set(specs) == {"search_policies", "get_policy", "get_case_status"}
    search = specs["search_policies"].parameters
    assert search["type"] == "object" and search["required"] == ["query"]
    assert "title" not in search


def test_search_returns_evidence(ctx: ToolContext) -> None:
    out, record = default_registry().execute(
        call("search_policies", query="call recordings retention"), ctx, 5
    )
    assert not out.is_error and out.content["results"]
    assert record.evidence_ids == [r["chunk_id"] for r in out.content["results"]]
    assert record.duration_ms >= 0


def test_search_respects_as_of_argument(ctx: ToolContext) -> None:
    out, _ = default_registry().execute(
        call("search_policies", query="proof of address", as_of="2024-01-01"), ctx, 5
    )
    assert out.content["as_of"] == "2024-01-01"
    assert all("@2.1" not in r["chunk_id"] for r in out.content["results"])


def test_get_policy_versions_and_acl(ctx: ToolContext) -> None:
    reg = default_registry()
    out, _ = reg.execute(call("get_policy", policy_id="pol-onb-021"), ctx, 5)
    assert out.content["version"] == "2.1" and len(out.content["versions"]) == 2
    assert out.evidence_ids[0] == "POL-ONB-021@2.1#s1"
    old, _ = reg.execute(call("get_policy", policy_id="POL-ONB-021", as_of="2023-07-01"), ctx, 5)
    assert old.content["version"] == "2.0" and old.content["superseded_on"] == "2024-03-01"
    restricted, _ = reg.execute(call("get_policy", policy_id="PRC-SAR-001"), ctx, 5)
    assert restricted.is_error and "restricted" in restricted.content["error"]
    missing, _ = reg.execute(call("get_policy", policy_id="POL-NOPE-1"), ctx, 5)
    assert missing.is_error
    early, _ = reg.execute(call("get_policy", policy_id="POL-ONB-021", as_of="2020-01-01"), ctx, 5)
    assert early.is_error and "in force" in early.content["error"]


def test_case_lookup(ctx: ToolContext) -> None:
    reg = default_registry()
    ok, _ = reg.execute(call("get_case_status", case_id="req-1001"), ctx, 5)
    assert ok.content["status"] == "pending-review" and ok.evidence_ids == ["CASE:REQ-1001"]
    missing, _ = reg.execute(call("get_case_status", case_id="REQ-9999"), ctx, 5)
    assert missing.is_error and "not found" in missing.content["error"]


def test_errors_become_results(ctx: ToolContext) -> None:
    reg = default_registry()
    unknown, rec = reg.execute(call("drop_tables"), ctx, 5)
    assert unknown.is_error and rec.is_error and "unknown tool" in unknown.content["error"]
    bad, _ = reg.execute(call("get_case_status", case_id="not a case"), ctx, 5)
    assert bad.is_error and "case_id" in bad.content["error"]
    bad_date, _ = reg.execute(call("search_policies", query="x y", as_of="yesterday"), ctx, 5)
    assert bad_date.is_error and "as_of" in bad_date.content["error"]


class _NoArgs(BaseModel):
    pass


def _slow(_: _NoArgs, __: ToolContext) -> ToolOutput:
    time.sleep(0.5)
    return ToolOutput(content={})


def _boom(_: _NoArgs, __: ToolContext) -> ToolOutput:
    raise RuntimeError("backend down")


def test_timeout_and_exception(ctx: ToolContext) -> None:
    reg = ToolRegistry([Tool("slow", "s", _NoArgs, _slow), Tool("boom", "b", _NoArgs, _boom)])
    assert reg.names == ["slow", "boom"]
    slow, _ = reg.execute(call("slow"), ctx, 0.05)
    assert slow.is_error and "timed out" in slow.content["error"]
    boom, _ = reg.execute(call("boom"), ctx, 1)
    assert boom.is_error and "backend down" in boom.content["error"]


def test_tool_context_is_request_scoped(ctx: ToolContext) -> None:
    assert ctx.as_of == date(2026, 10, 28) and ctx.role == "support"
