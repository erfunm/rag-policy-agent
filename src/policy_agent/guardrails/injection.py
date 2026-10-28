"""Prompt-injection defences.

1. Spotlighting: retrieved text is wrapped in ``<untrusted_document>`` delimiters, and the system
   prompt says that content is data, never instructions. Delimiter look-alikes inside the text are
   neutralised so a document cannot close its own wrapper.
2. Heuristic detection on the user question and on retrieved text. Retrieved sentences that look
   like instructions to an AI are removed before the model sees them (``strip_injections``).

Heuristics are a tripwire, not a guarantee. The real backstop is structural: the model can only
act through read-only tools, and every answer must cite evidence that the loop verifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instructions?|prompts?|rules|guidelines)\b",
            re.I,
        ),
    ),
    ("role_override", re.compile(r"\b(you are now|act as|pretend to be|from now on,? you)\b", re.I)),
    (
        "prompt_exfiltration",
        re.compile(r"\b(reveal|print|show|repeat|leak)\b[^.\n]{0,30}\b(system )?prompt\b", re.I),
    ),
    (
        "addressed_to_ai",
        re.compile(r"\b(note|message|instructions?) (to|for) (the )?(ai|llm|assistant|model)s?\b", re.I),
    ),
    ("output_override", re.compile(r"\b(reply|respond|answer|output) only with\b", re.I)),
    ("fake_delimiters", re.compile(r"</?\s*(system|assistant|untrusted_document|instructions?)\s*>", re.I)),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
REMOVED_MARKER = "[removed: possible prompt injection]"


def detect(text: str) -> list[str]:
    return [name for name, pattern in _RULES if pattern.search(text)]


@dataclass
class Sanitised:
    text: str
    flagged: list[str]


def sanitise(text: str, strip: bool = True) -> Sanitised:
    flagged: list[str] = []
    out: list[str] = []
    for para in text.split("\n"):
        sentences = _SENTENCE_SPLIT.split(para)
        kept: list[str] = []
        for s in sentences:
            hits = detect(s)
            flagged.extend(h for h in hits if h not in flagged)
            if hits and strip:
                if not kept or kept[-1] != REMOVED_MARKER:
                    kept.append(REMOVED_MARKER)
            else:
                kept.append(s)
        out.append(" ".join(kept))
    return Sanitised(text="\n".join(out), flagged=flagged)


def spotlight(doc_id: str, text: str) -> str:
    safe = re.sub(r"<\s*/?\s*untrusted_document", "&lt;untrusted_document", text, flags=re.I)
    return f'<untrusted_document id="{doc_id}">\n{safe}\n</untrusted_document>'
