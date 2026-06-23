"""Application-wide singletons (the RAG engine).

Kept in its own module so both the API routes and the tests can import and,
if needed, reset the engine without circular imports.
"""

from __future__ import annotations

from app.core.rag import RagEngine

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


def reset_engine() -> None:
    """Drop the current engine (used by tests for isolation)."""
    global _engine
    _engine = None
