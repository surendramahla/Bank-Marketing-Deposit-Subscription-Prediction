"""
core/config.py
--------------
Centralised configuration for the FastAPI ML Service.
All secrets are loaded from environment variables (.env file in development).
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "BankAI Pro - ML Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Paths ────────────────────────────────────────────────────────
    # Points to ml_service/models/ folder where .pkl files live
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    RAG_DOCS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag", "documents")
    VECTOR_STORE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag", "vector_store")

    # ── LLM Provider ─────────────────────────────────────────────────
    # Set LLM_PROVIDER=openai to switch to OpenAI GPT-4o
    LLM_PROVIDER: str = "gemini"  # "gemini" | "openai"
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Gemini model to use
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o"

    # ── LangChain ─────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""

    # ── CORS ─────────────────────────────────────────────────────────
    # Origins allowed to call this service
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",   # React frontend
        "http://localhost:5000",   # Node.js backend
        "http://localhost:5001",   # Original Flask app
        "http://localhost:8080",
    ]

    # ── RAG ──────────────────────────────────────────────────────────
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton of application settings."""
    return Settings()
