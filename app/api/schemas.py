"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int = Field(..., description="Number of chunks indexed from the document")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int | None = Field(None, ge=1, le=20)


class Citation(BaseModel):
    ref: int
    doc_id: str
    doc_name: str
    chunk_index: int
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str


class DocumentInfoModel(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedder: str
    llm: str
    provider: str
