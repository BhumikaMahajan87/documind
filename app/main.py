"""DocuMind FastAPI application entrypoint.

Run locally with:  uvicorn app.main:app --reload
Then open:          http://localhost:8000/docs   (interactive API)
                    http://localhost:8000/        (minimal web UI)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "RAG-based Document Q&A. Upload documents, then ask questions and get "
        "answers grounded in your documents with citations."
    ),
)

app.include_router(router, prefix="/api")

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>DocuMind</h1><p>See <a href='/docs'>/docs</a> for the API.</p>"
