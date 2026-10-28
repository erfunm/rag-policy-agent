from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from policy_agent.agent.loop import PolicyAgent
from policy_agent.config import RetrievalSettings, Settings, get_settings
from policy_agent.llm.base import LLMClient
from policy_agent.llm.fake import FakeLLM
from policy_agent.retrieval.bm25 import tokenize
from policy_agent.retrieval.dense import HashingEmbedder
from policy_agent.retrieval.hybrid import HybridRetriever, build_retriever
from policy_agent.tools.cases import CaseStore
from policy_agent.tools.registry import ToolContext

TODAY = date(2026, 10, 28)


class KeywordReranker:
    """Scores by query-word overlap: a deterministic stand-in for the cross-encoder in unit tests."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(t)))) - 1.0 for t in texts]


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.observability.audit_log_path = tmp_path / "audit.jsonl"
    return s


@pytest.fixture(scope="session")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> HybridRetriever:
    rs = RetrievalSettings(
        cache_dir=tmp_path_factory.mktemp("cache"), mode="hybrid_rerank", min_rerank_score=1.0
    )
    return build_retriever(rs, embedder=HashingEmbedder(), reranker=KeywordReranker())


@pytest.fixture(scope="session")
def cases() -> CaseStore:
    return CaseStore.from_file(Path("data/cases.json"))


@pytest.fixture
def ctx(retriever: HybridRetriever, cases: CaseStore, settings: Settings) -> ToolContext:
    return ToolContext(role="support", as_of=TODAY, retriever=retriever, cases=cases, settings=settings)


@pytest.fixture
def make_agent(retriever: HybridRetriever, cases: CaseStore, settings: Settings):  # type: ignore[no-untyped-def]
    def _make(llm: LLMClient | None = None, s: Settings | None = None) -> PolicyAgent:
        return PolicyAgent(settings=s or settings, llm=llm or FakeLLM(), retriever=retriever, cases=cases)

    return _make
