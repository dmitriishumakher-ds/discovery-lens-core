"""Chunk-level traceability index, built after clustering.

Pure stdlib. Maps every chunk_id to its text, origin and cluster assignment
so the results view can show the evidence behind each opportunity.
"""

from __future__ import annotations

from .models import Chunk, Cluster


def build_source_map(
    chunks: list[Chunk], clusters: list[Cluster]
) -> dict[str, dict[str, object]]:
    """chunk_id -> {text, filename, source_type, cluster_id}.
    cluster_id is None for any chunk not assigned to a cluster."""
    chunk_to_cluster: dict[str, int] = {
        chunk_id: cluster["cluster_id"]
        for cluster in clusters
        for chunk_id in cluster["all_chunk_ids"]
    }
    return {
        chunk["chunk_id"]: {
            "text": chunk["text"],
            "filename": chunk["filename"],
            "source_type": chunk["source_type"],
            "cluster_id": chunk_to_cluster.get(chunk["chunk_id"]),
        }
        for chunk in chunks
    }
