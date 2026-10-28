from datetime import date
from pathlib import Path

import pytest

from policy_agent.ingest.chunker import chunk_corpus, chunk_policy, index_text
from policy_agent.ingest.loader import PolicyFormatError, load_policies, parse_policy

POLICIES = Path("data/policies")

DOC = """---
policy_id: POL-X
title: Example
version: "1.0"
effective_date: 2024-01-01
owner: Owner
acl_roles: [support]
summary: s
---
# Example

## 1. First
Alpha text.

## 2. Second
{long}
"""


def test_parse_and_chunk_ids_are_stable() -> None:
    long = "\n\n".join(" ".join(["word"] * 50) for _ in range(4))
    doc = parse_policy(DOC.format(long=long), "x.md")
    chunks = chunk_policy(doc, max_words=120)
    assert [c.chunk_id for c in chunks] == ["POL-X@1.0#s1", "POL-X@1.0#s2.1", "POL-X@1.0#s2.2"]
    assert chunks[0].section == "First"
    assert index_text(chunks[0]).startswith("Example - First\n")
    assert [c.chunk_id for c in chunk_policy(doc, max_words=120)] == [c.chunk_id for c in chunks]


def test_doc_without_headings_is_one_chunk() -> None:
    doc = parse_policy(DOC.split("# Example")[0] + "Just a paragraph.", "y.md")
    assert [c.section for c in chunk_policy(doc)] == ["Overview"]


@pytest.mark.parametrize(
    "text",
    ["no front matter", "---\npolicy_id: X\n", "---\npolicy_id: X\n---\nbody"],
)
def test_bad_documents_raise(text: str) -> None:
    with pytest.raises(PolicyFormatError):
        parse_policy(text, "bad.md")


def test_corpus_versions_are_linked() -> None:
    docs = load_policies(POLICIES)
    onboarding = [d for d in docs if d.policy_id == "POL-ONB-021"]
    assert [d.version for d in onboarding] == ["2.0", "2.1"]
    assert onboarding[0].superseded_on == date(2024, 3, 1)
    assert onboarding[1].superseded_on is None
    assert onboarding[0].in_force(date(2024, 1, 1)) and not onboarding[1].in_force(date(2024, 1, 1))
    ids = [c.chunk_id for c in chunk_corpus(docs)]
    assert len(ids) == len(set(ids))
    assert any(d.acl_roles == ["compliance"] for d in docs)


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_policies(tmp_path)
