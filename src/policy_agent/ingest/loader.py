"""Load markdown policies with YAML front-matter and link versions of the same policy."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from policy_agent.models import PolicyDoc

_FRONT_MATTER_DELIM = "---"


class PolicyFormatError(ValueError):
    pass


def parse_policy(text: str, source: str) -> PolicyDoc:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIM:
        raise PolicyFormatError(f"{source}: missing YAML front-matter")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == _FRONT_MATTER_DELIM)
    except StopIteration as exc:
        raise PolicyFormatError(f"{source}: unterminated front-matter") from exc
    meta: dict[str, Any] = yaml.safe_load("\n".join(lines[1:end])) or {}
    body = "\n".join(lines[end + 1 :]).strip()
    try:
        return PolicyDoc(**meta, body=body, source=source)
    except Exception as exc:  # pydantic.ValidationError, TypeError on bad keys
        raise PolicyFormatError(f"{source}: {exc}") from exc


def link_versions(docs: list[PolicyDoc]) -> list[PolicyDoc]:
    """Set ``superseded_on`` on every version that a later version replaces."""
    by_id: dict[str, list[PolicyDoc]] = defaultdict(list)
    for doc in docs:
        by_id[doc.policy_id].append(doc)
    linked: list[PolicyDoc] = []
    for versions in by_id.values():
        versions.sort(key=lambda d: d.effective_date)
        for current, nxt in zip(versions, [*versions[1:], None], strict=True):
            linked.append(current.model_copy(update={"superseded_on": nxt.effective_date if nxt else None}))
    return sorted(linked, key=lambda d: (d.policy_id, d.effective_date))


def load_policies(directory: Path) -> list[PolicyDoc]:
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no policy markdown files in {directory}")
    return link_versions([parse_policy(p.read_text(encoding="utf-8"), p.name) for p in paths])
