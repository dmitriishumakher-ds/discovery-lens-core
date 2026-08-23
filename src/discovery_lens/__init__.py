"""Discovery Lens — turn qualitative discovery data into an evidence-backed
Opportunity-Solution Tree.

Public API surface. Heavy ML/LLM implementations live behind optional extras
and are injected into ``run_pipeline`` rather than imported at package load,
so importing ``discovery_lens`` stays cheap.
"""

from __future__ import annotations

from .config import PipelineConfig
from .models import ALLOWED_SOURCE_TYPES, Chunk, Cluster, ScoredCluster
from .pipeline import FileInput, PipelineError, PipelineResult, run_pipeline

__version__ = "2.0.0a0"

__all__ = [
    "ALLOWED_SOURCE_TYPES",
    "Chunk",
    "Cluster",
    "ScoredCluster",
    "PipelineConfig",
    "FileInput",
    "PipelineError",
    "PipelineResult",
    "run_pipeline",
    "__version__",
]
