"""Split raw text into overlapping chunks suitable for embedding.

Chunking matters in RAG: chunks that are too large dilute the embedding
signal, while chunks that are too small lose context. We split on sentence
boundaries first and then pack sentences into chunks that respect a target
character budget with a configurable overlap so context is not lost at the
seams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence-ending punctuation while keeping the delimiter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Pack sentences into overlapping chunks of roughly ``chunk_size`` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            # Start the next chunk with a tail of the previous one for context.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence

    if current:
        chunks.append(current)

    return [Chunk(text=c, index=i) for i, c in enumerate(chunks)]
