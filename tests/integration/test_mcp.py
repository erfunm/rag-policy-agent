import asyncio

import pytest

from policy_agent import mcp_server


@pytest.fixture
def server(retriever, cases, settings):  # type: ignore[no-untyped-def]
    return mcp_server.build_server(settings, retriever, cases, role="support")


def test_lists_three_tools(server) -> None:  # type: ignore[no-untyped-def]
    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == ["get_case_status", "get_policy", "search_policies"]
    search = next(t for t in tools if t.name == "search_policies")
    assert "query" in search.input_schema["properties"]


def test_call_tools(server) -> None:  # type: ignore[no-untyped-def]
    result = asyncio.run(server.call_tool("get_case_status", {"case_id": "REQ-1001"}))
    assert "pending-review" in str(result)
    restricted = asyncio.run(server.call_tool("get_policy", {"policy_id": "PRC-SAR-001"}))
    assert "restricted" in str(restricted)
    search = asyncio.run(server.call_tool("search_policies", {"query": "call recordings retention"}))
    assert "untrusted_document" in str(search)


def test_role_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_AGENT_MCP_ROLE", "compliance")
    assert mcp_server._role_from_env() == "compliance"
    monkeypatch.setenv("POLICY_AGENT_MCP_ROLE", "root")
    with pytest.raises(ValueError):
        mcp_server._role_from_env()


def test_lazy_backends(monkeypatch: pytest.MonkeyPatch, retriever, settings) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(mcp_server, "get_retriever", lambda _: retriever)
    server = mcp_server.build_server(settings, role="support")
    assert "approved" in str(asyncio.run(server.call_tool("get_case_status", {"case_id": "REQ-2002"})))
