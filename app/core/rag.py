"""The RAG engine: orchestrates ingestion, indexing, retrieval and answering.

This is the single object the API layer talks to. It owns the embedder, the
vector store and the LLM, and exposes two high-level operations:

* ``ingest`` - parse a document, chunk it, embed the chunks, and index them
* ``ask``    - embed the question, retrieve top-k chunks, and generate an answer
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.chunking import chunk_text
from app.core.embeddings import Embedder, build_embedder
from app.core.ingestion import extract_text
from app.core.llm import LLM, Answer, build_llm
from app.core.vector_store import StoredChunk, VectorStore


@dataclass
class IngestResult:
    doc_id: str
    doc_name: str
    chunks: int


@dataclass
class DocumentInfo:
    doc_id: str
    doc_name: str
    chunks: int


class RagEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder: Embedder = build_embedder(self.settings)
        self.store = VectorStore(dim=self.embedder.dim)
        self.llm: LLM = build_llm(self.settings)
        self._lock = threading.Lock()
        self._docs: dict[str, DocumentInfo] = {}

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        text = extract_text(filename, data)
        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document produced no extractable text")

        doc_id = uuid.uuid4().hex[:12]
        vectors = self.embedder.embed([c.text for c in chunks])
        stored = [
            StoredChunk(
                chunk_id=-1,
                doc_id=doc_id,
                doc_name=filename,
                chunk_index=c.index,
                text=c.text,
            )
            for c in chunks
        ]
        with self._lock:
            self.store.add(vectors, stored)
            self._docs[doc_id] = DocumentInfo(doc_id, filename, len(stored))

        return IngestResult(doc_id=doc_id, doc_name=filename, chunks=len(stored))

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        if not question.strip():
            raise ValueError("Question must not be empty")
        k = top_k or self.settings.top_k
        query_vec = self.embedder.embed([question])[0]
        with self._lock:
            results = self.store.search(query_vec, top_k=k)
        return self.llm.answer(question, results)

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            return list(self._docs.values())

    def stats(self) -> dict:
        return {
            "documents": len(self._docs),
            "chunks": self.store.size,
            "embedder": self.embedder.name,
            "llm": self.llm.name,
            "provider": self.settings.resolved_provider(),
        }
