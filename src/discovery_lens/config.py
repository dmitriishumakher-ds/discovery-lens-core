"""Pipeline configuration.

All tunable parameters live here, grouped per stage, instead of being
scattered as module-level constants across five files (v1). Frozen
dataclasses are used deliberately:

* immutability — a config object passed into the pipeline cannot be
  mutated mid-run, so a run is fully described by its config;
* zero dependencies — the core package needs nothing but the stdlib
  (pydantic-settings would add env-var parsing, but env handling belongs
  to the delivery layer: the API/UI builds a config and passes it down);
* testability — chunking strategy, weights and thresholds are parameters,
  not globals, so tests and the eval harness can sweep them.

Defaults reproduce the v1 production behaviour (T-07 token chunking,
T-09 HDBSCAN parameters, T-16-validated scoring weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "token"  # "token" (T-07 winner) | "sentence" (legacy fallback)
    tokens_per_chunk: int = 80
    token_overlap: int = 20
    min_chunk_tokens: int = 10  # trailing groups shorter than this are dropped
    # T-05 substantive-chunk filter
    min_sentences_per_chunk: int = 2
    min_tokens_per_chunk: int = 15
    # sentence strategy
    sentences_per_chunk: int = 3
    min_sentence_length: int = 10


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ClusteringConfig:
    random_state: int = 42
    min_chunks_for_hdbscan: int = 50
    # UMAP: 384 -> 5 dims so HDBSCAN can find density regions
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    # HDBSCAN
    hdbscan_min_cluster_size: int = 7
    hdbscan_min_samples: int = 3
    hdbscan_metric: str = "euclidean"
    hdbscan_selection_method: str = "eom"
    # noise reassignment: membership cap for chunks HDBSCAN refused
    noise_membership_cap: float = 0.5
    # output shape
    n_representative_chunks: int = 3
    n_boundary_chunks: int = 1


@dataclass(frozen=True)
class ScoringConfig:
    # evidence_robustness = diversity * diversity_weight + importance * size_weight
    diversity_weight: float = 0.65
    size_weight: float = 0.35
    # priority_score = (odi * odi_weight + evidence * evidence_weight) * dampening
    odi_weight: float = 0.60
    evidence_weight: float = 0.40
    goal_relevance_floor: float = 0.20  # D-03 dampening floor
    # D-02 recommendation quadrants
    odi_threshold: float = 0.10
    evidence_threshold: float = 0.40
    # sentiment
    sentiment_model: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    sentiment_batch_size: int = 8


@dataclass(frozen=True)
class LLMConfig:
    primary_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "llama-3.1-8b-instant"
    max_tokens: int = 4096
    system_prompt_path: Path = Path("prompts/system_prompt.txt")
    # clusters at or below this chunk count get jtbd_confidence forced to "low"
    min_cluster_size_for_confidence: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
