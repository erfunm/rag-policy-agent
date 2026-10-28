"""Dense retrieval: CPU ONNX embeddings via fastembed, cosine similarity in numpy, cached to disk."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...


def _normalise(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return _normalise(np.stack(list(self._model.passage_embed(list(texts)))).astype(np.float32))

    def embed_query(self, text: str) -> Matrix:
        return _normalise(np.stack(list(self._model.query_embed(text)))[0].astype(np.float32))


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Used in unit tests so they need no model download."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self._dim = dim

    def _embed(self, text: str) -> Matrix:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        return _normalise(v)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self._dim), np.float32)

    def embed_query(self, text: str) -> Matrix:
        return self._embed(text)


class DenseIndex:
    def __init__(self, texts: Sequence[str], embedder: Embedder, cache_dir: Path | None = None) -> None:
        self.embedder = embedder
        self.matrix = self._load_or_embed(texts, cache_dir)

    def _load_or_embed(self, texts: Sequence[str], cache_dir: Path | None) -> Matrix:
        if cache_dir is None:
            return self.embedder.embed_documents(texts)
        digest = hashlib.sha256("\x1e".join([self.embedder.name, *texts]).encode()).hexdigest()[:16]
        path = cache_dir / f"dense-{digest}.npy"
        if path.exists():
            cached: Matrix = np.load(path).astype(np.float32)
            return cached
        matrix = self.embedder.embed_documents(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    def scores(self, query: str) -> NDArray[np.float64]:
        return (self.matrix @ self.embedder.embed_query(query)).astype(np.float64)
