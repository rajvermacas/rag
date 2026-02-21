"""Shared pytest configuration for the project."""

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openrouter/test-chat")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv(
        "OPENROUTER_BATTLEGROUND_MODELS",
        "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet",
    )
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.4")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")
