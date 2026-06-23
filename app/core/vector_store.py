"""A small in-memory vector store with cosine-similarity search.

For a portfolio/demo project an in-memory store keeps the setup zero-config
while still demonstrating the core idea behind vector databases (FAISS,
Pinecone, pgvector, etc.): store embeddings and retrieve the nearest
neighbours for a query vector. Swapping this for a real vector DB later is a
matter of implementing the same ``add`` / ``search`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StoredChunk:
    chunk_id: int
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str


@dataclass
class SearchResult:
    chunk: StoredChunk
    score: float


@dataclass
class VectorStore:
    dim: int
    _vectors: list[np.ndarray] = field(default_factory=list)
    _chunks: list[StoredChunk] = field(default_factory=list)
    _next_id: int = 0

    def add(self, vectors: np.ndarray, chunks: list[StoredChunk]) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        for vec, chunk in zip(vectors, chunks):
            chunk.chunk_id = self._next_id
            self._next_id += 1
            self._vectors.append(vec.astype(np.float32))
            self._chunks.append(chunk)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if not self._vectors:
            return []
        matrix = np.vstack(self._vectors)
        # Vectors are pre-normalised, so the dot product is cosine similarity.
        scores = matrix @ query_vector.astype(np.float32)
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self._chunks}

    @property
    def size(self) -> int:
        return len(self._chunks)
