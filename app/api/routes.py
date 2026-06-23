"""API route definitions."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfoModel,
    IngestResponse,
    StatsResponse,
)
from app.core.ingestion import UnsupportedFileType
from app.deps import get_engine

router = APIRouter()

# Reject uploads larger than 10 MB to protect the in-memory store.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**get_engine().stats())


@router.post("/documents", response_model=IngestResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    try:
        result = get_engine().ingest(file.filename or "document", data)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(doc_id=result.doc_id, doc_name=result.doc_name, chunks=result.chunks)


@router.get("/documents", response_model=list[DocumentInfoModel])
def list_documents() -> list[DocumentInfoModel]:
    return [DocumentInfoModel(**d.__dict__) for d in get_engine().documents()]


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer = get_engine().ask(request.question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AskResponse(
        answer=answer.text,
        citations=answer.citations,
        provider=answer.provider,
    )
