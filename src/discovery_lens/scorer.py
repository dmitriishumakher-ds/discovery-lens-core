"""Deterministic cluster scoring. No LLM.

v2 changes vs v1:
* the sentiment model is injected as a plain callable
  (list[str] -> list[float] compounds in [-1, 1]) — unit tests and the
  eval harness pass a stub; production passes ``default_sentiment_fn``.
  In v1 the transformers pipeline was hard-wired into the module and
  decorated with Streamlit caching, which made the scorer untestable
  without loading a 250 MB model;
* score formulas are exposed as pure functions so weights/thresholds
  can be sensitivity-tested (T-16 style) directly;
* sentiment runs on torch (transformers default) — the v1 TensorFlow
  pin existed only to dodge an Apple-Silicon PyTorch bug and dragged a
  second ML framework into the image.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import ScoringConfig
from .models import ALLOWED_SOURCE_TYPES, Chunk, Cluster, ScoredCluster

logger = logging.getLogger(__name__)

SentimentFn = Callable[[list[str]], list[float]]

# ── Pure score formulas ───────────────────────────────────────────────────────


def compute_importance(cluster_size: int, total_chunks: int) -> float:
    """Share of the corpus this theme occupies. Range 0-1."""
    return cluster_size / total_chunks


def compute_satisfaction(avg_sentiment: float) -> float:
    """Rescale mean sentiment compound from [-1, 1] to [0, 1]."""
    return (avg_sentiment + 1) / 2


def compute_odi(importance: float, satisfaction: float) -> float:
    """Classic ODI unmet-need signal: importance x (1 - satisfaction)."""
    return importance * (1 - satisfaction)


def compute_evidence_robustness(
    source_type_diversity: float, importance: float, config: ScoringConfig
) -> float:
    return source_type_diversity * config.diversity_weight + importance * config.size_weight


def compute_priority(
    odi_score: float, evidence_robustness: float, goal_relevance: float, config: ScoringConfig
) -> float:
    """Synthesis with multiplicative goal-relevance dampening (D-03).
    The floor keeps off-goal clusters visible but ranked lower."""
    base = odi_score * config.odi_weight + evidence_robustness * config.evidence_weight
    return base * max(goal_relevance, config.goal_relevance_floor)


def recommendation_label(
    odi_score: float, evidence_robustness: float, config: ScoringConfig
) -> str:
    """D-02 quadrants: Act / Validate / Monitor / Deprioritise."""
    high_odi = odi_score >= config.odi_threshold
    high_evidence = evidence_robustness >= config.evidence_threshold
    if high_odi and high_evidence:
        return "Act"
    if high_odi:
        return "Validate"
    if high_evidence:
        return "Monitor"
    return "Deprioritise"


# ── Goal relevance ────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cluster_goal_relevance(
    chunk_ids: list[str],
    goal_embedding: np.ndarray | None,
    chunk_embedding_map: dict[str, np.ndarray],
    membership_scores: dict[str, float] | None = None,
) -> float:
    """Membership-weighted mean cosine similarity between the goal embedding
    and each chunk embedding in the cluster (D-03). Negative similarities are
    clipped to 0. Returns 1.0 (no dampening) when the goal embedding or the
    chunk embeddings are unavailable."""
    if goal_embedding is None or not chunk_embedding_map:
        return 1.0

    sims: list[float] = []
    weights: list[float] = []
    for chunk_id in chunk_ids:
        emb = chunk_embedding_map.get(chunk_id)
        if emb is None:
            continue
        sims.append(max(_cosine_similarity(goal_embedding, emb), 0.0))
        weight = 1.0
        if membership_scores:
            weight = float(membership_scores.get(chunk_id, 1.0))
        weights.append(weight)

    if not sims:
        return 1.0
    weights_arr = np.array(weights)
    total = weights_arr.sum()
    if total == 0:
        return float(np.mean(sims))
    return float(np.dot(sims, weights_arr) / total)


# ── Sentiment ────────────────────────────────────────────────────────────────


def default_sentiment_fn(config: ScoringConfig | None = None) -> SentimentFn:
    """Build the production sentiment callable (lazy transformers import).
    Returned function maps texts -> compound scores in [-1, 1]:
    positive -> +confidence, negative -> -confidence, neutral -> 0."""
    config = config or ScoringConfig()
    from transformers import pipeline as hf_pipeline  # heavy import, deferred

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=config.sentiment_model,
        truncation=True,
        max_length=512,
    )

    def _compound(result: dict) -> float:
        label = result["label"].lower()
        if label == "positive":
            return result["score"]
        if label == "negative":
            return -result["score"]
        return 0.0

    def sentiment_fn(texts: list[str]) -> list[float]:
        compounds: list[float] = []
        for i in range(0, len(texts), config.sentiment_batch_size):
            batch = texts[i : i + config.sentiment_batch_size]
            compounds.extend(_compound(r) for r in pipe(batch))
        return compounds

    return sentiment_fn


# ── Main scorer ──────────────────────────────────────────────────────────────


def score_clusters(
    clusters: list[Cluster],
    chunks: list[Chunk],
    goal_embedding: np.ndarray | None = None,
    chunk_embeddings: np.ndarray | None = None,
    *,
    sentiment_fn: SentimentFn | None = None,
    config: ScoringConfig | None = None,
) -> list[ScoredCluster]:
    """Score each cluster with four independent signals and a recommendation
    label. Sorted by priority_score descending. Raises ValueError on an
    empty chunk list."""
    config = config or ScoringConfig()
    if not chunks:
        raise ValueError("chunks list is empty — cannot compute scores")
    if sentiment_fn is None:
        sentiment_fn = default_sentiment_fn(config)

    chunk_text_map = {c["chunk_id"]: c["text"] for c in chunks}
    chunk_source_map = {c["chunk_id"]: c["source_type"] for c in chunks}

    chunk_embedding_map: dict[str, np.ndarray] = {}
    if chunk_embeddings is not None and goal_embedding is not None:
        for i, chunk in enumerate(chunks):
            if i < len(chunk_embeddings):
                chunk_embedding_map[chunk["chunk_id"]] = chunk_embeddings[i]

    total_chunks = len(chunks)
    scored: list[dict[str, Any]] = []

    for cluster in clusters:
        chunk_ids = cluster["all_chunk_ids"]
        cluster_size = len(chunk_ids)

        importance = compute_importance(cluster_size, total_chunks)

        texts = [chunk_text_map[cid] for cid in chunk_ids if cid in chunk_text_map]
        avg_sentiment = float(np.mean(sentiment_fn(texts))) if texts else 0.0
        satisfaction = compute_satisfaction(avg_sentiment)
        odi_score = compute_odi(importance, satisfaction)

        cluster_source_types = {
            chunk_source_map[cid] for cid in chunk_ids if cid in chunk_source_map
        }
        # Fixed denominator: diversity is scored against all recognised source
        # types regardless of what this session uploaded — stable across runs.
        source_type_diversity = len(cluster_source_types) / len(ALLOWED_SOURCE_TYPES)
        evidence_robustness = compute_evidence_robustness(
            source_type_diversity, importance, config
        )

        goal_relevance = cluster_goal_relevance(
            chunk_ids, goal_embedding, chunk_embedding_map,
            membership_scores=cluster.get("membership_scores"),
        )
        priority_score = compute_priority(odi_score, evidence_robustness, goal_relevance, config)

        scored.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_size": cluster_size,
                "importance": round(importance, 4),
                "avg_sentiment": round(avg_sentiment, 4),
                "satisfaction": round(satisfaction, 4),
                "source_type_diversity": round(source_type_diversity, 4),
                "odi_score": round(odi_score, 4),
                "evidence_robustness": round(evidence_robustness, 4),
                "goal_relevance": round(goal_relevance, 4),
                "priority_score": round(priority_score, 4),
                "recommendation": recommendation_label(odi_score, evidence_robustness, config),
            }
        )

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored  # type: ignore[return-value]
