"""PDF ingest adapter — extract text per page using pypdf."""
from __future__ import annotations

from pathlib import Path


def ingest_pdf(path: str | Path) -> list[str]:
    """Extract text from a PDF, one string per page.

    Args:
        path: filesystem path to a PDF file.

    Returns:
        list of page-strings; empty pages are preserved as empty strings so
        the index aligns with page numbers (page N == result[N-1]).
    """
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages
