"""Shared data contracts for the Discovery Lens pipeline.

Single source of truth for the source-type enum and the dict shapes that
flow between pipeline stages. v1 kept these as plain dicts documented in
docs/data_contracts.md; v2 formalises them as TypedDicts so mypy and IDEs
can catch contract violations, while staying wire-compatible with v1
(they are still plain dicts at runtime — no serialisation changes).
"""

from __future__ import annotations

from typing import TypedDict

# Recognised source types. Single definition — extractor, chunker and scorer
# all import from here (v1 duplicated this set in three modules).
ALLOWED_SOURCE_TYPES: tuple[str, ...] = (
    "interview",
    "review",
    "ticket",
    "usability",
    "social",
    "internal",
)


class Chunk(TypedDict):
    """Output of the chunker. chunks[i] pairs with embeddings[i] by index."""

    chunk_id: str
    text: str
    filename: str
    source_type: str


class Cluster(TypedDict):
    """Output of the clusterer."""

    cluster_id: int
    representative_chunks: list[Chunk]
    boundary_chunks: list[Chunk]
    all_chunk_ids: list[str]
    membership_scores: dict[str, float]


class ScoredCluster(TypedDict):
    """Output of the scorer. Sorted by priority_score descending."""

    cluster_id: int
    cluster_size: int
    importance: float
    avg_sentiment: float
    satisfaction: float
    source_type_diversity: float
    odi_score: float
    evidence_robustness: float
    goal_relevance: float
    priority_score: float
    recommendation: str
