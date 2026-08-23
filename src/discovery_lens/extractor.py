"""file -> raw text.

v2 changes vs v1: works on filesystem paths and raw bytes, not Streamlit
UploadedFile objects. The Streamlit UI adapter reads uploads to a temp path
(or passes bytes) and calls in here, so the pipeline no longer imports or
depends on any UI type. CSV extraction keeps only string columns and is
agnostic to column names/order — the four synthetic corpora each use a
different CSV schema, which the tests exercise directly.
"""

from __future__ import annotations

import io
from pathlib import Path

from .models import ALLOWED_SOURCE_TYPES


def extract_text(
    source: str | Path | bytes, source_type: str, *, filename: str | None = None
) -> str:
    """Extract plain text from a file path or raw bytes.

    Parameters
    ----------
    source:
        A filesystem path, or raw bytes (in which case ``filename`` must be
        given so the extension can be detected).
    source_type:
        One of ALLOWED_SOURCE_TYPES (fail fast on anything else).
    filename:
        Required when ``source`` is bytes; ignored otherwise.

    Returns "" for an empty file. Raises ValueError for an unknown
    source_type or unsupported extension.
    """
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type {source_type!r}. Must be one of: {sorted(ALLOWED_SOURCE_TYPES)}"
        )

    if isinstance(source, bytes):
        if filename is None:
            raise ValueError("filename is required when source is bytes")
        data = source
        name = filename.lower()
    else:
        path = Path(source)
        data = path.read_bytes()
        name = path.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(data)
    if name.endswith(".docx"):
        return _extract_docx(data)
    if name.endswith(".csv"):
        return _extract_csv(data)
    if name.endswith(".txt"):
        return _extract_txt(data)

    raise ValueError(f"Unsupported file extension: {name!r}. Supported: .pdf, .docx, .csv, .txt")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def _extract_csv(data: bytes) -> str:
    """Concatenate text-like columns row by row (' | ' between cells, newline
    between rows). Numeric columns (ratings, ids) are dropped as noise.
    Column names and order are irrelevant — only dtype matters."""
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data))
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        return ""
    lines: list[str] = []
    for _, row in df[text_cols].iterrows():
        cells = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace").strip()
