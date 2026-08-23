"""Pipeline orchestrator — the single entry point v1 never had.

In v1 the extract -> chunk -> embed -> cluster -> score -> build_ost chain
lived inside a Streamlit button handler (pages/upload.py), tangled with
session_state and progress spinners. That made it impossible to call the
pipeline from anything other than the UI — no tests, no API, no eval.

Here the chain is one framework-agnostic function. Delivery layers
(Streamlit today, FastAPI next) build inputs, call run_pipeline, and render
the result. The optional ``progress`` callback lets a UI report stage
transitions without the pipeline knowing what a UI is.

Embedding, clustering and LLM steps are injected as callables so tests and
the eval harness can run the full orchestration with lightweight stubs and
without loading any ML model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .chunker import chunk_text
from .config import PipelineConfig
from .models import Chunk, Cluster, ScoredCluster
from .scorer import SentimentFn, score_clusters
from .source_map import build_source_map

logger = logging.getLogger(__name__)


@dataclass
class FileInput:
    """One uploaded file, resolved to bytes by the delivery layer."""

    data: bytes
    filename: str
    source_type: str


@dataclass
class PipelineResult:
    chunks: list[Chunk]
    clusters: list[Cluster]
    scored_clusters: list[ScoredCluster]
    source_map: dict[str, dict[str, Any]]
    ost: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


# Injected-step signatures ------------------------------------------------------

class EmbedFn(Protocol):
    def __call__(self, chunks: list[Chunk]) -> np.ndarray: ...


class EmbedTextFn(Protocol):
    def __call__(self, text: str) -> np.ndarray: ...


class ClusterFn(Protocol):
    def __call__(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[Cluster]: ...


class BuildOstFn(Protocol):
    def __call__(
        self,
        clusters: list[Cluster],
        scored_clusters: list[ScoredCluster],
        goal: str,
        context_block: str,
    ) -> dict[str, Any]: ...


ProgressFn = Callable[[str], None]


def _noop_progress(_stage: str) -> None:
    pass


class PipelineError(RuntimeError):
    """Raised for expected, user-facing pipeline failures (no usable text,
    no clusters formed) so the delivery layer can show a clean message
    instead of a traceback."""


def run_pipeline(
    files: list[FileInput],
    goal: str,
    *,
    embed_fn: EmbedFn,
    embed_text_fn: EmbedTextFn,
    cluster_fn: ClusterFn,
    build_ost_fn: BuildOstFn,
    sentiment_fn: SentimentFn | None = None,
    context_block: str = "",
    config: PipelineConfig | None = None,
    progress: ProgressFn = _noop_progress,
) -> PipelineResult:
    """Run the full discovery pipeline on a set of files.

    The four ``*_fn`` steps are injected rather than imported so the heavy ML
    dependencies stay optional and the orchestration is unit-testable. The
    real implementations live in the ``ml`` and ``llm`` extras.
    """
    config = config or PipelineConfig()

    # 1. Extract + chunk --------------------------------------------------------
    progress("chunking")
    all_chunks: list[Chunk] = []
    for f in files:
        # Extraction is done by the caller for flexibility with bytes/paths;
        # here we chunk already-extracted text carried on FileInput.data.
        text = f.data.decode("utf-8", errors="replace") if isinstance(f.data, bytes) else f.data
        all_chunks.extend(
            chunk_text(text, f.filename, f.source_type, config=config.chunking)
        )
    if not all_chunks:
        raise PipelineError(
            "No usable text could be extracted. Check that the files contain readable text."
        )
    logger.info("run_pipeline: %d chunks from %d files", len(all_chunks), len(files))

    # 2. Embed ------------------------------------------------------------------
    progress("embedding")
    embeddings = embed_fn(all_chunks)

    # 3. Cluster ----------------------------------------------------------------
    progress("clustering")
    clusters = cluster_fn(all_chunks, embeddings)
    if not clusters:
        raise PipelineError(
            "No clusters could be formed. Try uploading more content across more source types."
        )

    # 4. Score ------------------------------------------------------------------
    progress("scoring")
    goal_embedding = embed_text_fn(goal) if goal else None
    scored = score_clusters(
        clusters,
        all_chunks,
        goal_embedding=goal_embedding,
        chunk_embeddings=embeddings,
        sentiment_fn=sentiment_fn,
        config=config.scoring,
    )

    # 5. Source map (traceability) ---------------------------------------------
    source_map = build_source_map(all_chunks, clusters)

    # 6. LLM -> OST -------------------------------------------------------------
    progress("framing")
    ost = build_ost_fn(clusters, scored, goal, context_block)

    return PipelineResult(
        chunks=all_chunks,
        clusters=clusters,
        scored_clusters=scored,
        source_map=source_map,
        ost=ost,
        meta={"n_files": len(files), "n_chunks": len(all_chunks), "n_clusters": len(clusters)},
    )
