from datetime import date
from pathlib import Path

import numpy as np
import pytest

from policy_agent.config import RetrievalSettings
from policy_agent.retrieval.bm25 import BM25Index, tokenize
from policy_agent.retrieval.dense import DenseIndex, HashingEmbedder
from policy_agent.retrieval.hybrid import HybridRetriever, build_retriever, rrf_fuse

TODAY = date(2026, 10, 28)


class KeywordReranker:
    """Scores by how many query words appear in the text: deterministic stand-in for a cross-encoder."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(t)))) - 1.0 for t in texts]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> HybridRetriever:
    settings = RetrievalSettings(cache_dir=tmp_path_factory.mktemp("cache"), mode="hybrid_rerank")
    return build_retriever(settings, embedder=HashingEmbedder(), reranker=KeywordReranker())


def test_rrf_fuse_rewards_agreement() -> None:
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_rrf_ties_break_by_index() -> None:
    assert [i for i, _ in rrf_fuse([[5], [2]])] == [2, 5]


def test_tokenize_keeps_amounts() -> None:
    assert tokenize("Above $5,000 per month, 85% match") == [
        "above",
        "$5,000",
        "per",
        "month",
        "85%",
        "match",
    ]


def test_bm25_prefers_matching_doc() -> None:
    idx = BM25Index(["sanctions screening of payments", "data retention for call recordings", "unrelated"])
    assert int(np.argmax(idx.scores("call recordings retention"))) == 1


def test_dense_cache_roundtrip(tmp_path: Path) -> None:
    first = DenseIndex(["a b", "c d"], HashingEmbedder(), tmp_path)
    assert len(list(tmp_path.glob("dense-*.npy"))) == 1
    second = DenseIndex(["a b", "c d"], HashingEmbedder(), tmp_path)
    np.testing.assert_allclose(first.matrix, second.matrix)
    assert DenseIndex([], HashingEmbedder()).matrix.shape == (0, 512)


def test_as_of_filter_selects_version(retriever: HybridRetriever) -> None:
    q = "proof of address monthly transaction volume"
    now = retriever.search(q, "support", TODAY)
    old = retriever.search(q, "support", date(2024, 1, 1))
    assert all(r.chunk.version == "2.1" for r in now.results if r.chunk.policy_id == "POL-ONB-021")
    assert all(r.chunk.version == "2.0" for r in old.results if r.chunk.policy_id == "POL-ONB-021")
    assert now.results[0].rerank_score is not None


def test_before_effective_date_nothing_in_force(retriever: HybridRetriever) -> None:
    assert retriever.search("proof of address", "support", date(2020, 1, 1)).results == []


def test_acl_filter_and_restricted_flag(retriever: HybridRetriever) -> None:
    q = "MLRO decision to file a suspicious activity report SAR FIU portal"
    support = retriever.search(q, "support", TODAY)
    assert all(r.chunk.visible_to("support") for r in support.results)
    assert support.access_restricted
    compliance = retriever.search(q, "compliance", TODAY)
    assert compliance.results[0].chunk.policy_id == "PRC-SAR-001"
    assert not compliance.access_restricted


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_rank_obvious_query(retriever: HybridRetriever, mode: str) -> None:
    res = retriever.search("call recordings retention months", "support", TODAY, mode=mode)  # type: ignore[arg-type]
    assert res.results[0].chunk.policy_id == "STD-DATA-014"
    assert [r.rank for r in res.results] == list(range(1, len(res.results) + 1))


def test_evidence_floor_uses_dense_without_rerank(retriever: HybridRetriever) -> None:
    res = retriever.search("zzz qqq", "support", TODAY, mode="hybrid")
    assert not res.evidence_sufficient
    assert retriever.policy_versions("POL-ONB-021")[0].version == "2.0"
