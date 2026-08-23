"""Chunk retrieval — the component Stage 2 will improve, defined in Stage 1.

Design notes:

* ``Retriever`` is a small protocol (index once, retrieve many) so the eval
  harness can score any implementation identically. Stage 2's hybrid
  retriever and reranker will plug into the same interface and be measured
  by the same harness — that is the whole point of building eval first.

* ``BM25Retriever`` is the lexical baseline. It is intentionally the first
  implementation: it is dependency-light (pure-python rank-bm25), runs in CI
  without model downloads, and its failure mode is exactly the phenomenon
  this corpus exhibits — the lexical gap. Users describe one event as
  "frozen", "blocked", "flagged", "on hold", "compliance review"; BM25 can
  only match the words the query happens to share.

* ``DenseRetriever`` wraps a sentence-transformers model (lazy import — the
  package must stay importable without the ML stack). Embeddings are
  L2-normalised so cosine similarity is a dot product.

* ``HybridRetriever`` fuses the two by rank. The baseline run showed three of
  twelve gold queries scoring zero on every metric — all three phrased
  abstractly, sharing no vocabulary with the chunks that answer them. That is
  the lexical gap measured rather than assumed, and fusion is the cheapest
  response to it that keeps BM25's precision on queries where exact wording
  does carry the signal.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from .models import Chunk


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score), best first."""
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25. Deliberately simple: no stemming,
    no stopwords — the baseline should be an honest, unembellished baseline."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lexical baseline over chunk texts (BM25 Okapi)."""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: list[str] = []

    def index(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi  # light, but keep core import-free

        self._chunk_ids = [c["chunk_id"] for c in chunks]
        corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Semantic retrieval via sentence-transformers embeddings.

    Requires the ``ml`` extra. Embeddings are normalised at encode time so
    ranking is a single matrix-vector product.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._chunk_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy, lazy

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, chunks: list[Chunk]) -> None:
        model = self._get_model()
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._matrix = model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("Call index() before retrieve()")
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [(self._chunk_ids[i], float(scores[i])) for i in order]


# Constant from Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet
# and Individual Rank Learning Methods" (SIGIR 2009). Left unchanged on
# purpose: tuning it against a 12-query gold set would fit noise, not signal.
RRF_K = 60

# Each retriever contributes this many candidates to the fusion, not just k.
# A chunk ranked 30th by one system and 2nd by the other should still be able
# to surface — that case is precisely what fusion exists to catch.
CANDIDATE_POOL = 50


class HybridRetriever:
    """Rank fusion of a lexical and a dense retriever (Reciprocal Rank Fusion).

    Why fuse ranks rather than scores: BM25 scores are unbounded and depend on
    corpus statistics, while cosine similarities live in [-1, 1]. A weighted
    sum of the two needs a normalisation that has to be re-tuned per corpus —
    an extra hyperparameter that buys nothing here. RRF looks only at the
    position a chunk occupies in each ranking, so the two systems become
    comparable without any calibration:

        score(chunk) = sum over retrievers of 1 / (RRF_K + rank)

    Both components are injected, so the harness can fuse any two retrievers
    satisfying the protocol — and tests can fuse stubs without loading a model.
    """

    def __init__(
        self,
        lexical: Retriever | None = None,
        dense: Retriever | None = None,
        *,
        rrf_k: int = RRF_K,
        candidate_pool: int = CANDIDATE_POOL,
    ) -> None:
        self.lexical: Retriever = lexical if lexical is not None else BM25Retriever()
        self.dense: Retriever = dense if dense is not None else DenseRetriever()
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, chunks: list[Chunk]) -> None:
        self.lexical.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        pool = max(self.candidate_pool, k)
        fused: dict[str, float] = {}

        for retriever in (self.lexical, self.dense):
            for rank, (chunk_id, _score) in enumerate(retriever.retrieve(query, pool), start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort by fused score, then by chunk_id: ties are common with RRF
        # (two chunks at mirrored ranks score identically) and an unstable
        # order would make eval results irreproducible run to run.
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]
