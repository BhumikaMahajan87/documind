"""Extract plain text from uploaded documents.

Supports PDF (via pypdf) and plain text / markdown. The extracted text is
normalised so that downstream chunking sees clean, single-spaced content.
"""

from __future__ import annotations

import io
import re


class UnsupportedFileType(ValueError):
    """Raised when an uploaded file type cannot be parsed."""


_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    """Return normalised text from the given file bytes."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _normalise(_extract_pdf(data))
    if name.endswith((".txt", ".md", ".markdown", ".rst")):
        return _normalise(data.decode("utf-8", errors="replace"))
    raise UnsupportedFileType(f"Unsupported file type: {filename}")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader  # lazy import keeps base import light

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
