"""Regex PII redaction with checksum validation, replacing matches with ``[REDACTED:type]``.

Card numbers must pass the Luhn check and IBANs the ISO 13616 mod-97 check, so policy amounts,
dates and case references are not redacted by accident.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


_Validator = Callable[[str], bool]

# Order matters: more specific patterns run first so e.g. an IBAN is not half-matched as a phone.
_PATTERNS: list[tuple[str, re.Pattern[str], _Validator]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), lambda _: True),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), iban_valid),
    ("card", re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])"), lambda m: luhn_valid(_digits(m))),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), lambda _: True),
    (
        "phone",
        re.compile(r"(?<![\w-])(?:\+\d{1,3}|0)[\d ()-]{7,16}\d(?![\w-])"),
        lambda m: 9 <= len(_digits(m)) <= 15,
    ),
]


@dataclass
class Redaction:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.found)


def redact(text: str) -> Redaction:
    result = Redaction(text=text)
    for kind, pattern, valid in _PATTERNS:

        def _sub(m: re.Match[str], kind: str = kind, valid: _Validator = valid) -> str:
            if not valid(m.group(0)):
                return m.group(0)
            result.found[kind] = result.found.get(kind, 0) + 1
            return f"[REDACTED:{kind}]"

        result.text = pattern.sub(_sub, result.text)
    return result
