"""raw text -> chunks.

v2 changes vs v1:
* chunking strategy is a config parameter, not a module-level flag —
  both paths are testable and the eval harness can A/B them;
* logging instead of print;
* nltk is imported lazily and only for the sentence strategy, so the
  default token path has zero third-party dependencies;
* terminology honesty: "tokens" here are whitespace-separated words.
  The T-07 benchmark (+31.6% silhouette vs sentence windows) was run
  with this word-window definition. A true-tokeniser variant is a
  planned eval-harness experiment, not a silent rename.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import ChunkingConfig
from .models import ALLOWED_SOURCE_TYPES, Chunk

logger = logging.getLogger(__name__)


def _safe_filename(filename: str) -> str:
    """'Interview 01.txt' -> 'interview_01' (chunk_id-safe slug)."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _is_chunk_substantive(text: str, config: ChunkingConfig) -> bool:
    """T-05 filter: single-sentence or sub-N-token chunks embed poorly and
    get absorbed by the nearest large cluster, inflating its importance."""
    fragments = [s.strip() for s in re.split(r"[.?!]+", text) if s.strip()]
    if len(fragments) < config.min_sentences_per_chunk:
        return False
    return len(text.split()) >= config.min_tokens_per_chunk


def _chunk_by_tokens(raw_text: str, config: ChunkingConfig) -> list[str]:
    """T-07 sliding word-window with overlap. Uniform chunk size gives more
    stable embeddings and tighter clusters than sentence windows."""
    tokens = raw_text.split()
    if len(tokens) < config.min_chunk_tokens:
        return []
    chunks: list[str] = []
    step = max(config.tokens_per_chunk - config.token_overlap, 1)
    for i in range(0, len(tokens), step):
        group = tokens[i : i + config.tokens_per_chunk]
        if len(group) < config.min_chunk_tokens:
            continue  # trailing group too short to carry semantic signal
        chunks.append(" ".join(group))
    return chunks


def _chunk_by_sentences(raw_text: str, config: ChunkingConfig) -> list[str]:
    """Legacy sentence-window strategy (pre-T-07). Kept as a benchmarkable
    fallback for corpora where sentence boundaries carry meaning."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The 'sentence' chunking strategy requires nltk. "
            "Install with: pip install 'discovery-lens[parsing]'"
        ) from e

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:  # pragma: no cover
        nltk.download("punkt_tab", quiet=True)

    sentences = [s.strip() for s in sent_tokenize(raw_text) if s.strip()]
    if not sentences:
        return []
    filtered = [s for s in sentences if len(s) >= config.min_sentence_length]
    sentences = filtered or sentences
    return [
        " ".join(sentences[i : i + config.sentences_per_chunk])
        for i in range(0, len(sentences), config.sentences_per_chunk)
    ]


def dedup_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """T-01: drop chunks with identical text (near-identical uploads under
    different filenames). Byte-level duplicates are caught upstream."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            unique.append(chunk)
    return unique


def chunk_text(
    raw_text: str,
    filename: str,
    source_type: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Split raw text into chunks with source metadata.

    Returns an empty list for empty input. Raises ValueError for an
    unknown source_type (fail fast — same rule as the extractor).
    """
    config = config or ChunkingConfig()

    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type {source_type!r}. Must be one of: {sorted(ALLOWED_SOURCE_TYPES)}"
        )
    if not raw_text or not raw_text.strip():
        return []

    if config.strategy == "token":
        chunk_strings = _chunk_by_tokens(raw_text, config)
    elif config.strategy == "sentence":
        chunk_strings = _chunk_by_sentences(raw_text, config)
    else:
        raise ValueError(f"Unknown chunking strategy {config.strategy!r}")

    if not chunk_strings:
        return []

    safe = _safe_filename(filename)
    chunks: list[Chunk] = [
        {
            "chunk_id": f"{safe}_{index:03d}",
            "text": text,
            "filename": filename,
            "source_type": source_type,
        }
        for index, text in enumerate(chunk_strings, start=1)
    ]

    substantive = [c for c in chunks if _is_chunk_substantive(c["text"], config)]
    dropped = len(chunks) - len(substantive)
    if dropped:
        logger.info(
            "chunker[%s]: T-05 filtered %d impoverished chunks (%d -> %d)",
            filename, dropped, len(chunks), len(substantive),
        )

    return dedup_chunks(substantive)
