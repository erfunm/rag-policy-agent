"""Contract tests for the real-provider adapters against a local stub HTTP server (no network, no key)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import anthropic
import openai
import pytest

from policy_agent.config import LLMSettings
from policy_agent.llm.anthropic import AnthropicClient, to_anthropic_messages
from policy_agent.llm.base import LLMError, Message, ToolCall, ToolSpec
from policy_agent.llm.openai_compat import OpenAICompatClient, to_openai_messages


class _Stub(BaseHTTPRequestHandler):
    responses: ClassVar[list[tuple[int, dict[str, Any]]]] = []
    requests: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        status, payload = type(self).responses.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def stub() -> Iterator[tuple[str, type[_Stub]]]:
    _Stub.responses, _Stub.requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _Stub
    server.shutdown()


TOOLS = [ToolSpec(name="get_case_status", description="d", parameters={"type": "object", "properties": {}})]
HISTORY = [
    Message(role="user", content="status of REQ-1001?"),
    Message(
        role="assistant",
        tool_calls=[ToolCall(id="t1", name="get_case_status", arguments={"case_id": "REQ-1001"})],
    ),
    Message(role="tool", tool_call_id="t1", content='{"status": "open"}'),
    Message(role="tool", tool_call_id="t2", content='{"error": "x"}', is_error=True),
]


def test_anthropic_request_and_response_shape(stub: tuple[str, type[_Stub]]) -> None:
    url, server = stub
    server.responses.append(
        (
            200,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_case_status",
                        "input": {"case_id": "REQ-1"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50},
            },
        )
    )
    client = AnthropicClient(LLMSettings(), anthropic.Anthropic(base_url=url, api_key="test", max_retries=0))
    turn = client.chat("system prompt", HISTORY, TOOLS)

    path, body = server.requests[0]
    assert path.startswith("/v1/messages")
    assert body["model"] == "claude-sonnet-5-5" and body["system"] == "system prompt"
    assert body["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "medium"}
    # both tool results merged into one user turn
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert [b["type"] for b in body["messages"][2]["content"]] == ["tool_result", "tool_result"]
    assert body["messages"][2]["content"][1]["is_error"] is True

    assert turn.text == "Checking." and turn.stop_reason == "tool_use"
    assert turn.tool_calls == [ToolCall(id="toolu_1", name="get_case_status", arguments={"case_id": "REQ-1"})]
    assert turn.usage.input_tokens == 150 and turn.usage.output_tokens == 20
    # provider-native content is replayed verbatim on the next request
    assert to_anthropic_messages([turn.as_message()])[0]["content"] is turn.raw


def test_anthropic_errors_become_llm_error(stub: tuple[str, type[_Stub]]) -> None:
    url, server = stub
    server.responses.append(
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
    )
    client = AnthropicClient(
        LLMSettings(anthropic_fallbacks=False, effort=None),
        anthropic.Anthropic(base_url=url, api_key="test", max_retries=0),
    )
    with pytest.raises(LLMError, match="400"):
        client.chat("s", [Message(role="user", content="hi")], TOOLS)
    assert "fallbacks" not in server.requests[0][1] and "output_config" not in server.requests[0][1]


def test_openai_request_and_response_shape(stub: tuple[str, type[_Stub]]) -> None:
    url, server = stub
    server.responses.append(
        (
            200,
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_case_status",
                                        "arguments": '{"case_id": "REQ-1"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_case_status", "arguments": "{oops"},
                                },
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37},
            },
        )
    )
    client = OpenAICompatClient(LLMSettings(openai_base_url=url), openai.OpenAI(base_url=url, api_key="t"))
    turn = client.chat("sys", HISTORY, TOOLS)
    path, body = server.requests[0]
    assert path == "/chat/completions"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "get_case_status"
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": '{"status": "open"}'}
    assert body["tools"][0]["function"]["parameters"] == TOOLS[0].parameters
    assert turn.stop_reason == "tool_use" and turn.usage.input_tokens == 30
    assert turn.tool_calls[0].arguments == {"case_id": "REQ-1"}
    assert "__unparseable__" in turn.tool_calls[1].arguments


def test_openai_errors_become_llm_error(stub: tuple[str, type[_Stub]]) -> None:
    url, server = stub
    server.responses.append((500, {"error": {"message": "down"}}))
    client = OpenAICompatClient(LLMSettings(), openai.OpenAI(base_url=url, api_key="t", max_retries=0))
    with pytest.raises(LLMError, match="500"):
        client.chat("s", [Message(role="user", content="hi")], TOOLS)


def test_message_converters_without_raw() -> None:
    msgs = [Message(role="assistant", content="hi", tool_calls=[ToolCall(id="a", name="n", arguments={})])]
    assert to_anthropic_messages(msgs)[0]["content"][1]["type"] == "tool_use"
    assert to_openai_messages("s", [Message(role="assistant", content="plain")])[1] == {
        "role": "assistant",
        "content": "plain",
    }


def test_make_llm_factory() -> None:
    from policy_agent.llm import make_llm

    assert make_llm(LLMSettings()).provider == "fake"
    assert make_llm(LLMSettings(), "openai").provider == "openai"
