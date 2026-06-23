"""Application configuration loaded from environment variables.

DocuMind works fully offline by default. If you provide a personal free
API key (OpenAI or Google Gemini) it will automatically use that provider
for higher-quality answers and embeddings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DocuMind"

    # Provider selection. "auto" picks the best available based on keys present.
    # Options: auto | offline | openai | gemini
    llm_provider: Literal["auto", "offline", "openai", "gemini"] = "auto"

    # Optional personal API keys. Leave blank to run fully offline.
    openai_api_key: str = ""
    gemini_api_key: str = ""

    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-flash"

    # Retrieval tuning.
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4

    def resolved_provider(self) -> str:
        """Resolve the concrete provider based on configuration + keys."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        return "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
