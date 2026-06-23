"""Pluggable answer generation backends.

Given a question and the retrieved context chunks, the LLM produces a final
answer. Three backends are supported and selected automatically:

* ``openai``  - chat completion via the OpenAI API (personal key)
* ``gemini``  - generation via the Google Gemini API (personal key)
* ``offline`` - an extractive answerer that needs no API key. It scores
  sentences in the retrieved context against the question and stitches the
  most relevant ones together. This guarantees the project runs end-to-end
  for anyone who clones it, with zero setup.

Every backend returns an answer plus the citations (which chunks were used),
which is the hallmark of a trustworthy RAG system.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings
from app.core.vector_store import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Answer:
    text: str
    citations: list[dict]
    provider: str


def _build_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] (source: {r.chunk.doc_name})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def _citations(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "ref": i,
            "doc_id": r.chunk.doc_id,
            "doc_name": r.chunk.doc_name,
            "chunk_index": r.chunk.chunk_index,
            "score": round(r.score, 4),
            "preview": r.chunk.text[:160] + ("..." if len(r.chunk.text) > 160 else ""),
        }
        for i, r in enumerate(results, start=1)
    ]


class LLM(ABC):
    @abstractmethod
    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class OfflineExtractiveLLM(LLM):
    """No-API-key answerer that extracts the most relevant sentences."""

    @property
    def name(self) -> str:
        return "offline-extractive"

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(
                text="I couldn't find anything relevant in the uploaded documents.",
                citations=[],
                provider=self.name,
            )

        q_tokens = set(_TOKEN_RE.findall(question.lower()))
        scored: list[tuple[float, int, str]] = []
        for ref, r in enumerate(results, start=1):
            for sentence in _SENTENCE_RE.split(r.chunk.text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                s_tokens = set(_TOKEN_RE.findall(sentence.lower()))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / (len(q_tokens) + 1e-9)
                # Blend lexical overlap with the chunk's retrieval score.
                score = overlap + 0.25 * r.score
                if overlap > 0:
                    scored.append((score, ref, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]

        if not top:
            # Fall back to the single best-matching chunk.
            best = results[0]
            text = f"Based on '{best.chunk.doc_name}': {best.chunk.text[:400]}"
            return Answer(text=text, citations=_citations(results), provider=self.name)

        sentences = [f"{sent} [{ref}]" for _, ref, sent in top]
        text = " ".join(sentences)
        return Answer(text=text, citations=_citations(results), provider=self.name)


_PROMPT = (
    "You are DocuMind, a helpful assistant. Answer the question using ONLY the "
    "context below. Cite sources inline using their bracket numbers like [1]. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        prompt = _PROMPT.format(context=_build_context(results), question=question)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        return Answer(text=text.strip(), citations=_citations(results), provider=self.name)


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str) -> None:
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    @property
    def name(self) -> str:
        return f"gemini/{self._model_name}"

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        prompt = _PROMPT.format(context=_build_context(results), question=question)
        resp = self._model.generate_content(prompt)
        return Answer(text=(resp.text or "").strip(), citations=_citations(results), provider=self.name)


def build_llm(settings: Settings) -> LLM:
    """Construct the best available LLM for the current configuration."""
    provider = settings.resolved_provider()
    try:
        if provider == "openai" and settings.openai_api_key:
            return OpenAILLM(settings.openai_api_key, settings.openai_model)
        if provider == "gemini" and settings.gemini_api_key:
            return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    except Exception:  # pragma: no cover - graceful fallback to offline
        pass
    return OfflineExtractiveLLM()
