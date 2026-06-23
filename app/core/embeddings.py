"""Pluggable embedding backends.

The embedder turns text into a dense vector so we can compare meaning via
cosine similarity. We support three tiers, picked automatically:

1. ``openai``  - uses OpenAI embeddings if a key is configured (best quality)
2. ``local``   - uses sentence-transformers if the package is installed
3. ``hashing`` - a dependency-free fallback so the app always runs offline

The hashing embedder uses the hashing trick with TF weighting and L2
normalisation. It is not as good as a neural model, but it is deterministic,
fast, and good enough to demonstrate the full RAG pipeline without any
external dependency or API key.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(ABC):
    """Abstract embedding backend."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised vectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class HashingEmbedder(Embedder):
    """Dependency-free embedder using the hashing trick + TF weighting."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec
        counts: dict[int, float] = {}
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            # Sign hashing reduces collisions cancelling each other out badly.
            sign = 1.0 if digest[4] & 1 else -1.0
            counts[bucket] = counts.get(bucket, 0.0) + sign
        for bucket, value in counts.items():
            # Opposite-sign collisions can cancel to exactly zero; skip those.
            if value == 0:
                continue
            # Sub-linear TF scaling dampens very frequent tokens.
            vec[bucket] = math.copysign(1 + math.log(abs(value)), value)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Local neural embedder via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI API (optional, needs a personal key)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = 1536

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def build_embedder(settings: Settings) -> Embedder:
    """Construct the best available embedder for the current configuration."""
    provider = settings.resolved_provider()

    if provider == "openai" and settings.openai_api_key:
        try:
            return OpenAIEmbedder(settings.openai_api_key)
        except Exception:  # pragma: no cover - falls back gracefully
            pass

    # Gemini path also benefits from a good local/offline embedder; try local.
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # pragma: no cover - optional dependency missing
        return HashingEmbedder()
