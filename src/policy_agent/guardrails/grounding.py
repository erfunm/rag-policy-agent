"""Output grounding check: an answer may only cite evidence retrieved in this request."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INLINE = re.compile(r"\[([A-Z]{2,5}[-:][A-Za-z0-9@#.:-]+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class GroundingResult:
    ok: bool
    citations: list[str]
    unknown: list[str] = field(default_factory=list)
    uncited_sentences: int = 0
    reason: str | None = None


def verify(answer: str, citations: list[str], evidence: list[str]) -> GroundingResult:
    """Check declared and inline citations against the evidence ids returned by tools.

    Fails if nothing is cited or any cited id was not retrieved. Sentences without an inline
    citation are counted (flagged, not failed), because short connective sentences are common.
    """
    inline = _INLINE.findall(answer)
    cited = list(dict.fromkeys([*citations, *inline]))
    unknown = [c for c in cited if c not in evidence]
    sentences = [s for s in _SENTENCE_SPLIT.split(answer.strip()) if s]
    uncited = sum(1 for s in sentences if not _INLINE.search(s))
    if not cited:
        return GroundingResult(ok=False, citations=[], uncited_sentences=uncited, reason="no citations")
    if unknown:
        return GroundingResult(
            ok=False,
            citations=cited,
            unknown=unknown,
            uncited_sentences=uncited,
            reason=f"unknown ids {unknown}",
        )
    return GroundingResult(ok=True, citations=cited, uncited_sentences=uncited)
