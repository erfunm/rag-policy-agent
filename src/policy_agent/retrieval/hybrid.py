"""Hybrid retrieval: BM25 + dense -> reciprocal rank fusion -> optional cross-encoder rerank.

Access control and the policy effective-date window are applied *before* ranking, so a chunk the
caller may not see, or a superseded version, can never be ranked, returned or cited.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from policy_agent.config import RetrievalMode, RetrievalSettings
from policy_agent.ingest.chunker import chunk_corpus, index_text
from policy_agent.ingest.loader import load_policies
from policy_agent.models import Chunk, PolicyDoc, RetrievedChunk, Role
from policy_agent.retrieval.bm25 import BM25Index
from policy_agent.retrieval.dense import DenseIndex, Embedder, FastEmbedEmbedder


class Reranker(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(texts))]


def rrf_fuse(rankings: Iterable[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _top(scores: NDArray[np.float64], candidates: NDArray[np.int64], n: int) -> list[int]:
    if candidates.size == 0:
        return []
    order = np.argsort(-scores[candidates], kind="stable")[:n]
    return [int(i) for i in candidates[order]]


@dataclass
class SearchResult:
    results: list[RetrievedChunk]
    # The best match for this query exists but is not visible to the caller's role.
    access_restricted: bool = False
    evidence_sufficient: bool = False
    top_relevance: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        docs: list[PolicyDoc],
        chunks: list[Chunk],
        dense: DenseIndex,
        settings: RetrievalSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self.docs = docs
        self.chunks = chunks
        self.settings = settings
        self.dense = dense
        self.reranker = reranker
        self.bm25 = BM25Index([index_text(c) for c in chunks])

    def _candidates(self, as_of: date, role: Role | None) -> NDArray[np.int64]:
        return np.asarray(
            [
                i
                for i, c in enumerate(self.chunks)
                if c.in_force(as_of) and (role is None or c.visible_to(role))
            ],
            dtype=np.int64,
        )

    def _rank(
        self,
        query: str,
        candidates: NDArray[np.int64],
        mode: RetrievalMode,
        sparse: NDArray[np.float64],
        dense: NDArray[np.float64],
    ) -> tuple[list[tuple[int, float]], bool]:
        """Return ranked (chunk index, score) pairs and whether the scores are cross-encoder scores."""
        pool = self.settings.candidate_pool
        if mode == "bm25":
            return [(i, float(sparse[i])) for i in _top(sparse, candidates, pool)], False
        if mode == "dense":
            return [(i, float(dense[i])) for i in _top(dense, candidates, pool)], False
        fused = rrf_fuse(
            [_top(sparse, candidates, pool), _top(dense, candidates, pool)], k=self.settings.rrf_k
        )
        fused = fused[:pool]
        if mode == "hybrid_rerank" and self.reranker is not None and fused:
            ce = self.reranker.rerank(query, [index_text(self.chunks[i]) for i, _ in fused])
            reranked = sorted(((i, s) for (i, _), s in zip(fused, ce, strict=True)), key=lambda kv: -kv[1])
            return reranked, True
        return fused, False

    def search(
        self,
        query: str,
        role: Role,
        as_of: date,
        k: int | None = None,
        mode: RetrievalMode | None = None,
    ) -> SearchResult:
        mode = mode or self.settings.mode
        k = k or self.settings.top_k
        sparse = self.bm25.scores(query)
        dense = self.dense.scores(query)

        ranked, reranked = self._rank(query, self._candidates(as_of, role), mode, sparse, dense)
        results = [
            RetrievedChunk(
                chunk=self.chunks[i],
                rank=r,
                score=s,
                relevance=float(dense[i]),
                rerank_score=s if reranked else None,
            )
            for r, (i, s) in enumerate(ranked[:k], start=1)
        ]
        top_rel = max((r.relevance for r in results), default=0.0)
        if reranked:
            sufficient = bool(results) and results[0].score >= self.settings.min_rerank_score
        else:
            sufficient = bool(results) and top_rel >= self.settings.min_relevance

        # Would an unrestricted search have put a hidden chunk first? Only the fact is disclosed,
        # never the content, so the agent can route the user to Compliance instead of guessing.
        unrestricted = rrf_fuse(
            [_top(sparse, self._candidates(as_of, None), 5), _top(dense, self._candidates(as_of, None), 5)],
            k=self.settings.rrf_k,
        )
        restricted = bool(unrestricted) and not self.chunks[unrestricted[0][0]].visible_to(role)

        return SearchResult(
            results=results,
            access_restricted=restricted,
            evidence_sufficient=sufficient,
            top_relevance=top_rel,
        )

    def policy_versions(self, policy_id: str) -> list[PolicyDoc]:
        return [d for d in self.docs if d.policy_id == policy_id]


def build_retriever(
    settings: RetrievalSettings, embedder: Embedder | None = None, reranker: Reranker | None = None
) -> HybridRetriever:
    docs = load_policies(settings.policies_dir)
    chunks = chunk_corpus(docs, settings.max_chunk_words)
    model_dir = settings.cache_dir / "models"
    embedder = embedder or FastEmbedEmbedder(settings.dense_model, model_dir)
    if reranker is None and settings.mode == "hybrid_rerank":
        reranker = FastEmbedReranker(settings.rerank_model, str(model_dir))
    dense = DenseIndex([index_text(c) for c in chunks], embedder, settings.cache_dir / "index")
    return HybridRetriever(docs, chunks, dense, settings, reranker)


@lru_cache(maxsize=4)
def get_retriever(settings: RetrievalSettings) -> HybridRetriever:
    """Process-wide retriever per settings value (models are loaded once)."""
    return build_retriever(settings)
