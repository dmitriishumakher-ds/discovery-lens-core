"""Retrieval evaluation: gold schema, quote->chunk mapping, metrics.

The key design decision here is what the gold labels point at.

Labelling chunk_ids directly would be fragile: chunk boundaries depend on
the chunking config, so the moment Stage 2 experiments with a different
window size or a real tokenizer, every label would silently point at the
wrong text. Instead, gold evidence is a *verbatim quote from a source file*.
At eval time each quote is resolved to whichever chunks contain it (after
whitespace/case normalisation). The gold set therefore survives any change
of chunking strategy — which is exactly the kind of change it exists to
measure.

A quote that resolves to zero chunks is treated as a loud error, not a
silent skip: it means either the quote is not verbatim or the chunker
dropped/split the passage, and both cases must be looked at by a human.

Metrics are standard binary-relevance IR metrics:

* recall@k — of all relevant chunks, what share appeared in the top k;
* MRR — reciprocal rank of the first relevant hit, averaged over queries.

Both are computed per query and macro-averaged, so a query with many
relevant chunks cannot dominate the aggregate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Chunk
from .retrieval import Retriever

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Whitespace-collapse + casefold, so quotes survive reflowing.
    Chunk text is whitespace-joined tokens, so this is the only
    normalisation needed for verbatim quotes."""
    return _WS_RE.sub(" ", text).casefold().strip()


# ── Gold schema ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoldEvidence:
    filename: str  # source file the quote comes from
    quote: str     # verbatim snippet (8-25 words is the sweet spot)


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    query: str          # PM-phrased information need
    rationale: str      # why this query is in the gold set
    evidence: tuple[GoldEvidence, ...]


def load_gold(path: str | Path) -> list[GoldQuery]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    queries: list[GoldQuery] = []
    for q in raw["queries"]:
        queries.append(
            GoldQuery(
                query_id=q["query_id"],
                query=q["query"],
                rationale=q.get("rationale", ""),
                evidence=tuple(
                    GoldEvidence(filename=e["filename"], quote=e["quote"])
                    for e in q["evidence"]
                ),
            )
        )
    return queries


# ── Quote -> chunk resolution ─────────────────────────────────────────────────


@dataclass
class ResolvedQuery:
    gold: GoldQuery
    relevant_chunk_ids: set[str]
    unmapped_quotes: list[GoldEvidence] = field(default_factory=list)


def resolve_gold_to_chunks(gold: list[GoldQuery], chunks: list[Chunk]) -> list[ResolvedQuery]:
    """Map every gold quote to the chunk(s) whose text contains it.

    A quote may legitimately map to two chunks (window overlap) — both are
    then relevant. Filename must match too, so a generic phrase cannot
    accidentally mark chunks from unrelated files as relevant.
    """
    norm_chunks = [
        (c["chunk_id"], c["filename"], normalize(c["text"])) for c in chunks
    ]
    resolved: list[ResolvedQuery] = []
    for gq in gold:
        relevant: set[str] = set()
        unmapped: list[GoldEvidence] = []
        for ev in gq.evidence:
            needle = normalize(ev.quote)
            hits = [
                cid
                for cid, fname, text in norm_chunks
                if fname == ev.filename and needle in text
            ]
            if hits:
                relevant.update(hits)
            else:
                unmapped.append(ev)
        resolved.append(
            ResolvedQuery(gold=gq, relevant_chunk_ids=relevant, unmapped_quotes=unmapped)
        )
    return resolved


# ── Metrics ───────────────────────────────────────────────────────────────────


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty — resolve gold before scoring")
    hits = sum(1 for cid in ranked[:k] if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


# ── Harness ───────────────────────────────────────────────────────────────────


def evaluate_retriever(
    retriever: Retriever,
    chunks: list[Chunk],
    gold: list[GoldQuery],
    ks: tuple[int, ...] = (5, 10),
    retrieve_depth: int | None = None,
) -> dict[str, Any]:
    """Index the corpus, run every gold query, return per-query and
    aggregate metrics plus data-quality warnings (unmapped quotes)."""
    depth = retrieve_depth or max(ks)
    resolved = resolve_gold_to_chunks(gold, chunks)

    retriever.index(chunks)

    per_query: list[dict[str, Any]] = []
    warnings: list[str] = []
    for rq in resolved:
        for ev in rq.unmapped_quotes:
            warnings.append(
                f"[{rq.gold.query_id}] quote not found in any chunk of "
                f"{ev.filename}: “{ev.quote[:60]}…”"
            )
        if not rq.relevant_chunk_ids:
            warnings.append(
                f"[{rq.gold.query_id}] SKIPPED — no evidence resolved to chunks"
            )
            continue

        ranked = [cid for cid, _score in retriever.retrieve(rq.gold.query, k=depth)]
        row: dict[str, Any] = {
            "query_id": rq.gold.query_id,
            "query": rq.gold.query,
            "n_relevant": len(rq.relevant_chunk_ids),
            "mrr": reciprocal_rank(rq.relevant_chunk_ids, ranked),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(rq.relevant_chunk_ids, ranked, k)
        per_query.append(row)

    n = len(per_query)
    aggregate: dict[str, float] = {}
    if n:
        aggregate["mrr"] = sum(r["mrr"] for r in per_query) / n
        for k in ks:
            aggregate[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / n

    return {
        "n_queries": n,
        "n_chunks": len(chunks),
        "aggregate": aggregate,
        "per_query": per_query,
        "warnings": warnings,
    }
