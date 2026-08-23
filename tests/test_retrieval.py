"""Tests for retrieval — fusion logic only, no models.

The point of the ``Retriever`` protocol is that fusion can be tested against
stubs: these tests run in CI in milliseconds and never download a model.
"""

from __future__ import annotations

import pytest

from discovery_lens.retrieval import HybridRetriever


class StubRetriever:
    """Returns a fixed ranking, ignoring the query. Satisfies the protocol."""

    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking
        self.indexed: list[dict] | None = None

    def index(self, chunks: list[dict]) -> None:
        self.indexed = chunks

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        # Descending scores so the stub looks like a real ranking.
        return [(cid, 1.0 / (n + 1)) for n, cid in enumerate(self.ranking)][:k]


def test_consensus_beats_a_single_first_place():
    """The core property of RRF: a chunk both retrievers rank moderately well
    outranks chunks that only one retriever puts first. This is why fusion
    helps on queries where the two systems disagree."""
    lexical = StubRetriever(["a", "b", "shared", "c"])
    dense = StubRetriever(["x", "y", "shared", "z"])

    hybrid = HybridRetriever(lexical, dense)
    hybrid.index([])

    ranked = [chunk_id for chunk_id, _ in hybrid.retrieve("any query", k=4)]

    assert ranked[0] == "shared"
    assert set(ranked[1:3]) == {"a", "x"}


def test_ties_are_broken_deterministically():
    """Mirrored ranks produce identical fused scores. Without a stable
    tiebreak the eval numbers would drift between runs on the same data."""
    hybrid = HybridRetriever(StubRetriever(["a", "b"]), StubRetriever(["b", "a"]))
    hybrid.index([])

    first = hybrid.retrieve("q", k=2)
    assert first == hybrid.retrieve("q", k=2)
    assert [chunk_id for chunk_id, _ in first] == ["a", "b"]


def test_k_larger_than_the_union_returns_the_union():
    hybrid = HybridRetriever(StubRetriever(["a", "b"]), StubRetriever(["c"]))
    hybrid.index([])

    assert len(hybrid.retrieve("q", k=99)) == 3


def test_index_reaches_both_components():
    lexical, dense = StubRetriever(["a"]), StubRetriever(["b"])
    chunks = [{"chunk_id": "a", "text": "..."}]

    HybridRetriever(lexical, dense).index(chunks)

    assert lexical.indexed is chunks
    assert dense.indexed is chunks


def test_scores_are_descending():
    hybrid = HybridRetriever(StubRetriever(["a", "b", "c"]), StubRetriever(["c", "b", "a"]))
    hybrid.index([])

    scores = [score for _, score in hybrid.retrieve("q", k=3)]

    assert scores == sorted(scores, reverse=True)


@pytest.fixture
def agree_deep() -> tuple[StubRetriever, StubRetriever]:
    """Both retrievers rank ``deep`` 21st — agreed on, but far past k."""
    lexical = StubRetriever([f"filler{n}" for n in range(20)] + ["deep"])
    dense = StubRetriever([f"other{n}" for n in range(20)] + ["deep"])
    return lexical, dense


def test_wide_candidate_pool_lets_an_agreed_deep_hit_surface(agree_deep):
    """A chunk both retrievers rank deep is still the strongest consensus in
    the corpus and must reach the final top-k. This is why each retriever is
    asked for ``candidate_pool`` results rather than ``k``."""
    hybrid = HybridRetriever(*agree_deep, candidate_pool=50)
    hybrid.index([])

    assert "deep" in [chunk_id for chunk_id, _ in hybrid.retrieve("q", k=3)]


def test_narrow_candidate_pool_never_sees_it(agree_deep):
    """Mirror of the test above: clamp the pool to k and the fusion never
    sees the chunk at all. Documents why the default is 50, not k."""
    hybrid = HybridRetriever(*agree_deep, candidate_pool=3)
    hybrid.index([])

    assert "deep" not in [chunk_id for chunk_id, _ in hybrid.retrieve("q", k=3)]


def test_zero_k_returns_nothing():
    hybrid = HybridRetriever(StubRetriever(["a", "b"]), StubRetriever(["b", "a"]))
    hybrid.index([])

    assert hybrid.retrieve("q", k=0) == []
