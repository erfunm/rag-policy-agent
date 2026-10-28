"""Deterministic offline LLMs.

``FakeLLM`` is a rule-based stand-in that speaks the same tool-calling protocol as the real
adapters: it plans tool calls from the question, reads the tool results from the transcript and
composes an extractive, cited answer through ``submit_answer``. It lets tests, CI and the demo run
with no API key, and it makes the retrieval and refusal metrics reproducible.

``ScriptedLLM`` replays a fixed list of turns, for unit-testing the loop's edge cases.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from policy_agent.llm.base import LLMTurn, Message, ToolCall, ToolSpec
from policy_agent.models import Usage
from policy_agent.retrieval.bm25 import tokenize

_CASE_ID = re.compile(r"\b[A-Z]{3}-\d{3,6}\b")
_QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_UNTRUSTED = re.compile(r"<untrusted_document[^>]*>\s*(.*?)\s*</untrusted_document>", re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_POLICY_INTENT = re.compile(
    r"\b(policy|procedure|require|required|must|should|next|deadline|how long|rule)", re.I
)
# Score gap (cross-encoder logits) within which a second policy is also cited.
_SECOND_SOURCE_MARGIN = 2.5


@dataclass
class _Result:
    name: str
    arguments: dict[str, Any]
    content: dict[str, Any]
    is_error: bool


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _question(messages: Sequence[Message]) -> str:
    first = next((m.content for m in messages if m.role == "user"), "")
    match = _QUESTION.search(first)
    return match.group(1) if match else first


def _results(messages: Sequence[Message]) -> list[_Result]:
    calls = {c.id: c for m in messages if m.role == "assistant" for c in m.tool_calls}
    out: list[_Result] = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id in calls:
            call = calls[m.tool_call_id]
            try:
                content = json.loads(m.content)
            except json.JSONDecodeError:
                content = {"error": m.content}
            out.append(_Result(call.name, call.arguments, content, m.is_error))
    return out


def _best_sentences(text: str, question: str, n: int) -> list[str]:
    inner = _UNTRUSTED.search(text)
    body = inner.group(1) if inner else text
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if s.strip() and "[removed" not in s.lower()]
    q = set(tokenize(question))
    ranked = sorted(range(len(sentences)), key=lambda i: (-len(q & set(tokenize(sentences[i]))), i))
    return [sentences[i] for i in sorted(ranked[:n])]


class FakeLLM:
    provider = "fake"
    model = "fake-llm"

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        step = sum(1 for m in messages if m.role == "assistant") + 1
        question = _question(messages)
        results = _results(messages)
        calls = self._plan(question, step) if not results else [self._answer(question, results, step)]
        prompt_tokens = _estimate_tokens(system + "".join(m.content for m in messages))
        out_tokens = _estimate_tokens(json.dumps([c.arguments for c in calls]))
        return LLMTurn(
            tool_calls=calls,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=out_tokens),
            stop_reason="tool_use",
            model=self.model,
        )

    def _plan(self, question: str, step: int) -> list[ToolCall]:
        case_ids = _CASE_ID.findall(question.upper())
        calls = [
            ToolCall(id=f"fake_{step}_{i}", name="get_case_status", arguments={"case_id": cid})
            for i, cid in enumerate(case_ids)
        ]
        if not case_ids or _POLICY_INTENT.search(_CASE_ID.sub("", question)):
            query = _CASE_ID.sub("", question).strip() or question
            calls.append(
                ToolCall(id=f"fake_{step}_{len(calls)}", name="search_policies", arguments={"query": query})
            )
        return calls

    def _answer(self, question: str, results: list[_Result], step: int) -> ToolCall:
        def submit(
            status: str, answer: str, citations: list[str], reason: str | None, confidence: str
        ) -> ToolCall:
            args = {"status": status, "answer": answer, "citations": citations, "confidence": confidence}
            if reason:
                args["reason"] = reason
            return ToolCall(id=f"fake_{step}_submit", name="submit_answer", arguments=args)

        def escalate(reason: str) -> ToolCall:
            return submit("escalated", "Escalating to Compliance.", [], reason, "low")

        parts: list[str] = []
        citations: list[str] = []
        confidence = "medium"
        for r in results:
            if r.is_error:
                return escalate(str(r.content.get("error", "tool error")))
            if r.name == "get_case_status":
                c = r.content
                parts.append(
                    f"Case {c['case_id']} is {c['status']} ({c['type']}; risk flag: {c['risk_flag']}), "
                    f"assigned to {c['assigned_team']}. [{c['evidence_id']}]"
                )
                citations.append(c["evidence_id"])
        for r in results:
            if r.name != "search_policies":
                continue
            c = r.content
            if c["access_restricted"]:
                return escalate("the best match is in a policy restricted for this role")
            if not c["evidence_sufficient"] or not c["results"]:
                if not parts:
                    return escalate("insufficient evidence in the policy library")
                continue
            hits = c["results"]
            top = hits[0]
            confidence = "high" if top["score"] >= 3.0 else "medium"
            picks = [(top, 2)]
            second = next((h for h in hits[1:3] if h["policy_id"] != top["policy_id"]), None)
            if second is not None and second["score"] >= top["score"] - _SECOND_SOURCE_MARGIN:
                picks.append((second, 1))
            for hit, n in picks:
                for sentence in _best_sentences(hit["text"], question, n):
                    parts.append(f"{sentence} [{hit['chunk_id']}]")
                citations.append(hit["chunk_id"])
        if not parts:
            return escalate("no usable evidence")
        return submit("answered", " ".join(parts), citations, None, confidence)


class ScriptedLLM:
    """Returns pre-defined turns in order; records what it was sent."""

    provider = "scripted"
    model = "scripted-llm"

    def __init__(self, turns: Sequence[LLMTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[Message]] = []

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        self.calls.append(list(messages))
        if not self._turns:
            return LLMTurn(text="I have nothing more to add.", stop_reason="end_turn", model=self.model)
        return self._turns.pop(0)
