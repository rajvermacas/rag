"""Shared pytest configuration for the project."""

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm,azure_prod")

    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_PROVIDER", "openrouter")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_MODELS", "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_API_KEY", "test-chat-key")

    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_PROVIDER", "azure_openai")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_MODELS", "gpt-4o-mini")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_API_KEY", "test-azure-key")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_AZURE_ENDPOINT", "https://azure-openai.example.com")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_AZURE_DEPLOYMENTS", "gpt-4o-mini=chat-gpt4o-mini")

    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.4")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")
