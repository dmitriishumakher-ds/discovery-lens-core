"""Unit tests for the evaluation harness and the BM25 baseline."""

from __future__ import annotations

import pytest

from discovery_lens.evaluation import (
    GoldEvidence,
    GoldQuery,
    normalize,
    recall_at_k,
    reciprocal_rank,
    resolve_gold_to_chunks,
)
from discovery_lens.retrieval import BM25Retriever


def _chunk(cid, text, filename="f.txt"):
    return {"chunk_id": cid, "text": text, "filename": filename, "source_type": "interview"}


# ── normalize ─────────────────────────────────────────────────────────────────

def test_normalize_collapses_whitespace_and_case():
    assert normalize("  My  Card\n got   Blocked ") == "my card got blocked"


# ── metrics ───────────────────────────────────────────────────────────────────

def test_recall_at_k_basic():
    relevant = {"a", "b", "c", "d"}
    ranked = ["x", "a", "b", "y", "z"]
    assert recall_at_k(relevant, ranked, 5) == pytest.approx(0.5)  # 2 of 4
    assert recall_at_k(relevant, ranked, 2) == pytest.approx(0.25)  # only "a"


def test_recall_empty_relevant_raises():
    with pytest.raises(ValueError):
        recall_at_k(set(), ["a"], 5)


def test_reciprocal_rank():
    assert reciprocal_rank({"b"}, ["a", "b", "c"]) == pytest.approx(0.5)
    assert reciprocal_rank({"a"}, ["a"]) == pytest.approx(1.0)
    assert reciprocal_rank({"zzz"}, ["a", "b"]) == 0.0


# ── quote -> chunk resolution ─────────────────────────────────────────────────

def test_resolve_finds_quote_despite_whitespace_and_case():
    text = "My card GOT   blocked when I was in Croatia actually and it was scary."
    chunks = [_chunk("c1", text)]
    gold = [
        GoldQuery(
            query_id="q1",
            query="card blocking abroad",
            rationale="",
            evidence=(
                GoldEvidence(filename="f.txt", quote="my card got blocked when I was in Croatia"),
            ),
        )
    ]
    resolved = resolve_gold_to_chunks(gold, chunks)
    assert resolved[0].relevant_chunk_ids == {"c1"}
    assert resolved[0].unmapped_quotes == []


def test_resolve_respects_filename_scoping():
    # same phrase in a different file must NOT count as relevant
    chunks = [_chunk("c1", "the exact same phrase here", filename="other.txt")]
    gold = [
        GoldQuery(
            query_id="q1",
            query="q",
            rationale="",
            evidence=(GoldEvidence(filename="f.txt", quote="the exact same phrase here"),),
        )
    ]
    resolved = resolve_gold_to_chunks(gold, chunks)
    assert resolved[0].relevant_chunk_ids == set()
    assert len(resolved[0].unmapped_quotes) == 1


def test_resolve_maps_quote_to_multiple_overlapping_chunks():
    # window overlap can put the same quote in two chunks — both are relevant
    chunks = [
        _chunk("c1", "intro words my card got blocked tail words"),
        _chunk("c2", "my card got blocked and then more trailing context here"),
    ]
    gold = [
        GoldQuery(
            query_id="q1",
            query="q",
            rationale="",
            evidence=(GoldEvidence(filename="f.txt", quote="my card got blocked"),),
        )
    ]
    resolved = resolve_gold_to_chunks(gold, chunks)
    assert resolved[0].relevant_chunk_ids == {"c1", "c2"}


# ── BM25 sanity ───────────────────────────────────────────────────────────────

def test_bm25_ranks_lexical_match_first():
    chunks = [
        _chunk("about_cats", "cats are wonderful pets that purr and sleep all day"),
        _chunk("about_banks", "the bank froze my account during a compliance review"),
        _chunk("about_food", "this restaurant serves excellent pasta and pizza"),
    ]
    r = BM25Retriever()
    r.index(chunks)
    top = r.retrieve("account frozen compliance", k=3)
    assert top[0][0] == "about_banks"


def test_bm25_misses_lexical_gap():
    """Documents the baseline's known weakness: 'blocked' does not match
    'frozen'. This is the exact failure mode Stage 2's hybrid search must
    fix — and this test pins the behaviour we expect to beat."""
    chunks = [
        _chunk("blocked", "my card got blocked at the airport and I was powerless"),
        _chunk("weather", "the weather in Lisbon is sunny and warm in October"),
    ]
    r = BM25Retriever()
    r.index(chunks)
    scores = dict(r.retrieve("frozen account", k=2))
    # BM25 gives the truly-relevant 'blocked' chunk a zero score:
    assert scores.get("blocked", 0.0) == 0.0


def test_bm25_requires_index_first():
    with pytest.raises(RuntimeError):
        BM25Retriever().retrieve("anything")
