import json

import pytest
from fastapi.testclient import TestClient

from policy_agent.api.app import create_app


@pytest.fixture
def client(make_agent):  # type: ignore[no-untyped-def]
    with TestClient(create_app(make_agent())) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["provider"] == "fake" and body["chunks"] > 40


def test_ask_json(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "call recordings retention", "as_of": "2026-10-28"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"] and body["request_id"]


def test_role_header_drives_acl(client: TestClient) -> None:
    q = {
        "question": "MLRO decision to file a suspicious activity report SAR FIU portal",
        "as_of": "2026-10-28",
    }
    support = client.post("/ask", json=q, headers={"X-User-Role": "support"}).json()
    compliance = client.post("/ask", json=q, headers={"X-User-Role": "compliance"}).json()
    assert support["status"] == "escalated"
    assert compliance["status"] == "answered" and compliance["citations"][0].startswith("PRC-SAR-001")


def test_bad_role_and_validation(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "x"}, headers={"X-User-Role": "admin"}).status_code == 400
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_stream_sse(client: TestClient) -> None:
    with client.stream("POST", "/ask/stream", json={"question": "status of REQ-1001"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [block.split("\n") for block in text.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[0] == "start" and names[-1] == "final" and "tool_call" in names
    final = json.loads(events[-1][1].removeprefix("data: "))
    assert final["citations"] == ["CASE:REQ-1001"]


def test_openapi_documents_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/ask", "/ask/stream", "/healthz"} <= set(paths)
