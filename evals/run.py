"""Entry point for the eval harness: ``uv run python evals/run.py`` == ``uv run policy-agent eval``."""

import sys

from policy_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
