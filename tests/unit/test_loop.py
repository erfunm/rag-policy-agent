import json

from policy_agent.config import Settings
from policy_agent.llm.base import LLMError, LLMTurn, Message, ToolCall, ToolSpec
from policy_agent.llm.fake import FakeLLM, ScriptedLLM
from policy_agent.models import AskRequest, Usage
from tests.conftest import TODAY


def tool_turn(*calls: tuple[str, dict[str, object]], usage: int = 10) -> LLMTurn:
    return LLMTurn(
        tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=usage, output_tokens=usage),
        model="claude-sonnet-5-5",
    )


def submit(**kw: object) -> tuple[str, dict[str, object]]:
    return ("submit_answer", {"status": "answered", "answer": "x", "confidence": "high", **kw})


ASK = AskRequest(question="How long are call recordings kept?", as_of=TODAY)


def test_happy_path_with_real_tool_round_trip(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(
        [
            tool_turn(("search_policies", {"query": "call recordings retention"})),
            tool_turn(submit(citations=["STD-DATA-014@1.4#s2"], answer="12 months [STD-DATA-014@1.4#s2]")),
        ]
    )
    resp = make_agent(llm).ask(ASK)
    assert resp.status == "answered" and resp.citations == ["STD-DATA-014@1.4#s2"]
    assert resp.steps == 2 and resp.usage.input_tokens == 20
    assert resp.cost_usd > 0  # priced from settings.llm.prices
    tool_msg = llm.calls[1][-1]
    assert tool_msg.role == "tool" and json.loads(tool_msg.content)["results"]


def test_hallucinated_citation_is_downgraded(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(
        [
            tool_turn(("search_policies", {"query": "call recordings"})),
            tool_turn(submit(citations=["POL-MADE-UP@1#s1"])),
        ]
    )
    resp = make_agent(llm).ask(ASK)
    assert resp.status == "escalated" and "grounding_failed" in resp.flags
    assert resp.answer == Settings().agent.escalation_message


def test_answer_without_citations_is_downgraded(make_agent) -> None:  # type: ignore[no-untyped-def]
    resp = make_agent(ScriptedLLM([tool_turn(submit(citations=[]))])).ask(ASK)
    assert resp.status == "escalated" and "grounding_failed" in resp.flags


def test_tool_error_goes_back_to_model(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(
        [
            tool_turn(("get_case_status", {"case_id": "REQ-9999"})),
            tool_turn(("submit_answer", {"status": "escalated", "answer": "", "reason": "case not found"})),
        ]
    )
    resp = make_agent(llm).ask(ASK)
    assert llm.calls[1][-1].is_error
    assert resp.status == "escalated" and resp.reason == "case not found"
    assert resp.tool_calls[0].is_error


def test_invalid_submit_is_reported_and_retried(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(
        [
            tool_turn(("submit_answer", {"status": "maybe"})),
            tool_turn(("search_policies", {"query": "call recordings"})),
            tool_turn(submit(citations=["STD-DATA-014@1.4#s2"])),
        ]
    )
    resp = make_agent(llm).ask(ASK)
    assert "invalid submit_answer" in llm.calls[1][-1].content
    assert resp.status == "answered"


def test_submit_with_other_calls_is_deferred(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM(
        [
            tool_turn(
                ("search_policies", {"query": "call recordings"}), submit(citations=["STD-DATA-014@1.4#s2"])
            ),
            tool_turn(submit(citations=["STD-DATA-014@1.4#s2"])),
        ]
    )
    resp = make_agent(llm).ask(ASK)
    assert resp.status == "answered" and resp.steps == 2


def test_plain_text_gets_one_nudge_then_escalates(make_agent) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedLLM([LLMTurn(text="It is 12 months."), LLMTurn(text="Really, 12 months.")])
    resp = make_agent(llm).ask(ASK)
    assert resp.status == "escalated" and resp.reason == "no_structured_answer"
    assert "submit_answer" in llm.calls[1][-1].content


def test_step_limit(make_agent, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    settings.agent.max_steps = 2
    llm = ScriptedLLM([tool_turn(("search_policies", {"query": "loop"})) for _ in range(5)])
    resp = make_agent(llm, settings).ask(ASK)
    assert resp.status == "escalated" and resp.reason == "step_limit_reached" and resp.steps == 2


def test_refusal_and_llm_error_escalate(make_agent) -> None:  # type: ignore[no-untyped-def]
    refused = make_agent(ScriptedLLM([LLMTurn(stop_reason="refusal")])).ask(ASK)
    assert refused.reason == "model_refusal"

    class Broken:
        provider = "broken"
        model = "broken"

        def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMTurn:
            raise LLMError("503")

    failed = make_agent(Broken()).ask(ASK)
    assert failed.status == "escalated" and failed.reason is not None and "503" in failed.reason


def test_stream_emits_events(make_agent) -> None:  # type: ignore[no-untyped-def]
    events = list(make_agent(FakeLLM()).stream(ASK))
    types = [e.type for e in events]
    assert types[0] == "start" and types[-1] == "final"
    assert "tool_call" in types and "tool_result" in types


def test_fake_llm_end_to_end(make_agent) -> None:  # type: ignore[no-untyped-def]
    agent = make_agent(FakeLLM())
    current = agent.ask(AskRequest(question="proof of address monthly transaction volume", as_of=TODAY))
    assert current.status == "answered" and "$5,000" in current.answer
    assert current.citations[0].startswith("POL-ONB-021@2.1")
    assert not any("@2.0" in c for c in current.citations)
    case = agent.ask(AskRequest(question="What is the status of case REQ-1001?"))
    assert case.status == "answered" and case.citations == ["CASE:REQ-1001"]
    missing = agent.ask(AskRequest(question="status of REQ-9999"))
    assert missing.status == "escalated"
    restricted = agent.ask(
        AskRequest(question="MLRO decision to file a suspicious activity report SAR FIU portal", as_of=TODAY)
    )
    assert (
        restricted.status == "escalated"
        and restricted.reason is not None
        and "restricted" in restricted.reason
    )
    oos = agent.ask(AskRequest(question="zzz qqq", as_of=TODAY))
    assert oos.status == "escalated"
