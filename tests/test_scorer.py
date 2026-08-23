"""Unit tests for the scorer. Sentiment is stubbed — no model loads."""

from __future__ import annotations

import numpy as np
import pytest

from discovery_lens.config import ScoringConfig
from discovery_lens.scorer import (
    cluster_goal_relevance,
    compute_evidence_robustness,
    compute_importance,
    compute_odi,
    compute_priority,
    compute_satisfaction,
    recommendation_label,
    score_clusters,
)

# ── Pure formulas ─────────────────────────────────────────────────────────────

def test_importance_is_corpus_share():
    assert compute_importance(20, 100) == 0.2


def test_satisfaction_rescales_sentiment():
    assert compute_satisfaction(-1.0) == 0.0
    assert compute_satisfaction(0.0) == 0.5
    assert compute_satisfaction(1.0) == 1.0


def test_odi_high_when_important_and_unsatisfied():
    # large cluster (0.4) with low satisfaction (0.1) -> high unmet need
    assert compute_odi(0.4, 0.1) == pytest.approx(0.36)


def test_evidence_robustness_weights():
    cfg = ScoringConfig()
    # diversity 1.0, importance 0.0 -> just the diversity weight
    assert compute_evidence_robustness(1.0, 0.0, cfg) == pytest.approx(0.65)
    assert compute_evidence_robustness(0.0, 1.0, cfg) == pytest.approx(0.35)


def test_priority_dampening_floor_applies():
    cfg = ScoringConfig()
    # goal_relevance below floor is clamped to the floor (0.20), never zeroed
    low = compute_priority(0.5, 0.5, goal_relevance=0.0, config=cfg)
    floored = compute_priority(0.5, 0.5, goal_relevance=0.20, config=cfg)
    assert low == pytest.approx(floored)
    assert low > 0


@pytest.mark.parametrize(
    "odi,evidence,expected",
    [
        (0.20, 0.50, "Act"),
        (0.20, 0.30, "Validate"),
        (0.05, 0.50, "Monitor"),
        (0.05, 0.30, "Deprioritise"),
        (0.10, 0.40, "Act"),  # thresholds are inclusive
    ],
)
def test_recommendation_quadrants(odi, evidence, expected):
    assert recommendation_label(odi, evidence, ScoringConfig()) == expected


# ── Goal relevance ────────────────────────────────────────────────────────────

def test_goal_relevance_none_embedding_returns_one():
    assert cluster_goal_relevance(["a"], None, {}) == 1.0


def test_goal_relevance_identical_vectors_is_one():
    goal = np.array([1.0, 0.0, 0.0])
    emb_map = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([1.0, 0.0, 0.0])}
    assert cluster_goal_relevance(["a", "b"], goal, emb_map) == pytest.approx(1.0)


def test_goal_relevance_orthogonal_is_zero():
    goal = np.array([1.0, 0.0])
    emb_map = {"a": np.array([0.0, 1.0])}
    assert cluster_goal_relevance(["a"], goal, emb_map) == pytest.approx(0.0)


def test_goal_relevance_membership_weighting():
    goal = np.array([1.0, 0.0])
    # one on-goal chunk (sim 1) with high weight, one off-goal (sim 0) low weight
    emb_map = {"hi": np.array([1.0, 0.0]), "lo": np.array([0.0, 1.0])}
    weighted = cluster_goal_relevance(
        ["hi", "lo"], goal, emb_map, membership_scores={"hi": 0.9, "lo": 0.1}
    )
    unweighted = cluster_goal_relevance(["hi", "lo"], goal, emb_map)
    assert weighted > unweighted  # weighting toward the on-goal chunk raises it


# ── Full scorer with stubbed sentiment ────────────────────────────────────────

def _make_chunks(n, source_type="review", prefix="c"):
    return [
        {
            "chunk_id": f"{prefix}_{i:03d}", "text": f"text {i}",
            "filename": "f", "source_type": source_type,
        }
        for i in range(n)
    ]


def test_score_clusters_sorted_by_priority():
    chunks = _make_chunks(10)
    clusters = [
        {"cluster_id": 0, "representative_chunks": [], "boundary_chunks": [],
         "all_chunk_ids": [c["chunk_id"] for c in chunks[:8]], "membership_scores": {}},
        {"cluster_id": 1, "representative_chunks": [], "boundary_chunks": [],
         "all_chunk_ids": [c["chunk_id"] for c in chunks[8:]], "membership_scores": {}},
    ]
    # all-negative sentiment -> low satisfaction -> ODI driven by size
    scored = score_clusters(
        clusters, chunks, sentiment_fn=lambda texts: [-1.0] * len(texts)
    )
    assert [s["cluster_id"] for s in scored] == [0, 1]  # bigger cluster ranks first
    assert scored[0]["priority_score"] >= scored[1]["priority_score"]


def test_score_clusters_diversity_uses_fixed_denominator():
    # a cluster spanning 3 of 6 source types -> diversity 0.5
    chunks = (
        _make_chunks(2, "review", "r")
        + _make_chunks(2, "interview", "i")
        + _make_chunks(2, "ticket", "t")
    )
    all_ids = [c["chunk_id"] for c in chunks]
    clusters = [
        {"cluster_id": 0, "representative_chunks": [], "boundary_chunks": [],
         "all_chunk_ids": all_ids, "membership_scores": {}},
    ]
    scored = score_clusters(clusters, chunks, sentiment_fn=lambda texts: [0.0] * len(texts))
    assert scored[0]["source_type_diversity"] == pytest.approx(0.5)


def test_score_clusters_empty_chunks_raises():
    with pytest.raises(ValueError, match="empty"):
        score_clusters([], [], sentiment_fn=lambda t: [])
