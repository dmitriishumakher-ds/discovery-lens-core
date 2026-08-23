"""Unit tests for the chunker — pure logic, no ML dependencies."""

from __future__ import annotations

import pytest

from discovery_lens.chunker import (
    _chunk_by_tokens,
    _is_chunk_substantive,
    _safe_filename,
    chunk_text,
    dedup_chunks,
)
from discovery_lens.config import ChunkingConfig

# ── _safe_filename ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Interview 01.txt", "interview_01"),
        ("reviews_revolut.csv", "reviews_revolut"),
        ("UPPER-CASE__Name.md", "upper_case_name"),
        ("weird!!!chars@@@.txt", "weird_chars"),
    ],
)
def test_safe_filename(raw, expected):
    assert _safe_filename(raw) == expected


# ── _is_chunk_substantive ─────────────────────────────────────────────────────

def test_substantive_rejects_single_short_sentence():
    cfg = ChunkingConfig()
    assert _is_chunk_substantive("Yeah, ok.", cfg) is False


def test_substantive_accepts_multi_sentence_long_chunk():
    cfg = ChunkingConfig()
    text = (
        "The checkout process is slow and confusing. I could not find where to "
        "change my payment method. Support never replied to my ticket."
    )
    assert _is_chunk_substantive(text, cfg) is True


def test_substantive_rejects_enough_sentences_but_too_few_tokens():
    cfg = ChunkingConfig()
    # two sentences, but only 4 tokens total
    assert _is_chunk_substantive("Yes. No.", cfg) is False


# ── _chunk_by_tokens ──────────────────────────────────────────────────────────

def test_token_window_overlap():
    cfg = ChunkingConfig(tokens_per_chunk=10, token_overlap=3, min_chunk_tokens=3)
    text = " ".join(str(i) for i in range(25))  # 25 tokens
    chunks = _chunk_by_tokens(text, cfg)
    # step = 10 - 3 = 7 -> windows start at 0, 7, 14, 21
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "7"
    # overlap: last 3 tokens of window 0 reappear at the start of window 1
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_token_window_drops_short_trailing_group():
    cfg = ChunkingConfig(tokens_per_chunk=10, token_overlap=0, min_chunk_tokens=5)
    text = " ".join(str(i) for i in range(23))  # 23 tokens -> 10,10,3
    chunks = _chunk_by_tokens(text, cfg)
    # trailing group of 3 (< min_chunk_tokens=5) is dropped
    assert len(chunks) == 2


def test_token_window_empty_below_minimum():
    cfg = ChunkingConfig(min_chunk_tokens=10)
    assert _chunk_by_tokens("one two three", cfg) == []


# ── dedup ─────────────────────────────────────────────────────────────────────

def test_dedup_removes_identical_text():
    chunks = [
        {"chunk_id": "a_001", "text": "same text here", "filename": "a", "source_type": "review"},
        {"chunk_id": "b_001", "text": "same text here", "filename": "b", "source_type": "review"},
        {"chunk_id": "c_001", "text": "different", "filename": "c", "source_type": "review"},
    ]
    result = dedup_chunks(chunks)  # type: ignore[arg-type]
    assert len(result) == 2
    assert result[0]["chunk_id"] == "a_001"  # first occurrence wins


# ── chunk_text (integration of the above) ─────────────────────────────────────

def test_chunk_text_rejects_unknown_source_type():
    with pytest.raises(ValueError, match="Invalid source_type"):
        chunk_text("some text", "f.txt", "not_a_type")


def test_chunk_text_empty_input_returns_empty():
    assert chunk_text("", "f.txt", "review") == []
    assert chunk_text("   \n  ", "f.txt", "review") == []


def test_chunk_text_produces_padded_ids():
    cfg = ChunkingConfig(tokens_per_chunk=10, token_overlap=2, min_chunk_tokens=3,
                         min_sentences_per_chunk=1, min_tokens_per_chunk=1)
    text = ". ".join(f"sentence number {i} with several words in it" for i in range(20))
    chunks = chunk_text(text, "interview_01.txt", "interview", config=cfg)
    assert chunks[0]["chunk_id"] == "interview_01_001"
    assert all(c["source_type"] == "interview" for c in chunks)


def test_chunk_text_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("a b c d e f", "f.txt", "review", config=ChunkingConfig(strategy="bogus"))
