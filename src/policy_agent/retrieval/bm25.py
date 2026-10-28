"""Sparse lexical retrieval (BM25 Okapi) over chunk index text."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\$?[a-z0-9]+(?:[.,][0-9]+)*%?")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i if in is it its of on or our that the their "
    "this to was we what when where which who why will with do does can should must".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, texts: Sequence[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> NDArray[np.float64]:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
