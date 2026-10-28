"""Command line: ``policy-agent ask | ingest``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import get_args

from policy_agent.config import Provider, get_settings
from policy_agent.models import ROLES, AgentResponse, AskRequest

_STATUS_COLOUR = {"answered": "\033[32m", "escalated": "\033[33m"}


def _print_response(resp: AgentResponse, colour: bool) -> None:
    c, reset = (_STATUS_COLOUR[resp.status], "\033[0m") if colour else ("", "")
    print(
        f"{c}[{resp.status.upper()}]{reset} confidence={resp.confidence}"
        + (f" reason={resp.reason}" if resp.reason else "")
    )
    print()
    print(resp.answer)
    print()
    if resp.citations:
        print("Citations: " + ", ".join(resp.citations))
    for t in resp.tool_calls:
        args = json.dumps(t.arguments, ensure_ascii=False)
        print(f"  tool {t.name}({args}) {'ERROR' if t.is_error else 'ok'} {t.duration_ms:.0f}ms")
    if resp.flags:
        print("Flags: " + ", ".join(resp.flags))
    print(
        f"model={resp.model} prompt={resp.prompt_version} steps={resp.steps} "
        f"tokens={resp.usage.input_tokens}/{resp.usage.output_tokens} cost=${resp.cost_usd:.5f} "
        f"latency={resp.latency_ms:.0f}ms request_id={resp.request_id}"
    )


def cmd_ask(args: argparse.Namespace) -> int:
    from policy_agent.agent.loop import build_agent
    from policy_agent.llm import make_llm

    settings = get_settings()
    agent = build_agent(settings, make_llm(settings.llm, args.llm))
    resp = agent.ask(AskRequest(question=args.question, role=args.role, as_of=args.as_of))
    if args.json:
        print(resp.model_dump_json(indent=2))
    else:
        _print_response(resp, sys.stdout.isatty())
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from policy_agent.retrieval.hybrid import build_retriever

    r = build_retriever(get_settings().retrieval)
    policies = {d.policy_id for d in r.docs}
    print(f"indexed {len(r.chunks)} chunks from {len(r.docs)} policy versions ({len(policies)} policies)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask a policy question")
    ask.add_argument("question")
    ask.add_argument("--role", choices=ROLES, default="support")
    ask.add_argument("--as-of", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)")
    ask.add_argument("--llm", choices=get_args(Provider), default=None, help="override settings.llm.provider")
    ask.add_argument("--json", action="store_true", help="print the full structured response")
    ask.set_defaults(fn=cmd_ask)

    ingest = sub.add_parser("ingest", help="load, chunk and embed the policy corpus (cached)")
    ingest.set_defaults(fn=cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
