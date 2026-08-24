"""
Centralized configuration for Railway RAG Assistant.

All environment variables are validated at import time using Pydantic Settings.
Usage:
    from app.config import settings
    print(settings.GEMINI_MODEL)
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # LLM Provider
    LLM_PROVIDER: str = Field(default="gemini", description="gemini, lmstudio, or openrouter")

    # Google Gemini
    GOOGLE_API_KEY: str = Field(default="", description="Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-3.6-flash", description="Gemini model name")

    # LM Studio (local)
    LOCAL_API_BASE: str = Field(default="http://localhost:1234/v1")
    LOCAL_MODEL_NAME: str = Field(default="google/gemma-2-9b")

    # Embeddings — always Gemini (gemini-embedding-001, 3072 dims)
    # HuggingFace / local models are not used.

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"  # ignore unknown env vars

    @property
    def is_gemini(self) -> bool:
        return self.LLM_PROVIDER.lower().strip() == "gemini"

    @property
    def has_api_key(self) -> bool:
        return bool(self.GOOGLE_API_KEY) and self.GOOGLE_API_KEY != "your-gemini-api-key-here"


# Singleton — validated at import time
settings = Settings()
