"""Heading-aware chunking with stable, human-readable chunk ids.

Each ``## `` section becomes one chunk (``POL-ONB-021@2.1#s3``). A section longer than
``max_words`` is split on paragraph boundaries into ``#s3.1``, ``#s3.2`` ... so ids only change
when the document structure changes, not when an unrelated section is edited.
"""

from __future__ import annotations

import re

from policy_agent.models import Chunk, PolicyDoc

_HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_NUMBER_PREFIX = re.compile(r"^\d+(\.\d+)*\.?\s+")


def _split_sections(body: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("Overview", body.strip())] if body.strip() else []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end() : end].strip()
        if text:
            sections.append((_NUMBER_PREFIX.sub("", m.group("title")), text))
    return sections


def _split_long(text: str, max_words: int) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if current and len(" ".join([*current, para]).split()) > max_words:
            parts.append("\n\n".join(current))
            current = []
        current.append(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_policy(doc: PolicyDoc, max_words: int = 180) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s_idx, (section, text) in enumerate(_split_sections(doc.body), start=1):
        pieces = _split_long(text, max_words)
        for p_idx, piece in enumerate(pieces, start=1):
            suffix = f"s{s_idx}" if len(pieces) == 1 else f"s{s_idx}.{p_idx}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.key}#{suffix}",
                    policy_id=doc.policy_id,
                    version=doc.version,
                    title=doc.title,
                    section=section,
                    text=piece,
                    effective_date=doc.effective_date,
                    superseded_on=doc.superseded_on,
                    acl_roles=doc.acl_roles,
                )
            )
    return chunks


def chunk_corpus(docs: list[PolicyDoc], max_words: int = 180) -> list[Chunk]:
    return [c for d in docs for c in chunk_policy(d, max_words)]


def index_text(chunk: Chunk) -> str:
    """Text used for indexing: the chunk plus a contextual header so short sections stay findable."""
    return f"{chunk.title} - {chunk.section}\n{chunk.text}"
