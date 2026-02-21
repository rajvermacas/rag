import os

import pytest

from app.config import Settings, load_environment_from_dotenv


def test_missing_required_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openrouter/test-chat")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.75")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")

    with pytest.raises(
        ValueError, match="Missing required environment variable: OPENROUTER_API_KEY"
    ):
        Settings.from_env()


def test_invalid_integer_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openrouter/test-chat")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "bad-int")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.75")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")

    with pytest.raises(
        ValueError, match="Invalid integer for environment variable MAX_UPLOAD_MB"
    ):
        Settings.from_env()


def test_settings_from_env_success(required_env: None) -> None:
    settings = Settings.from_env()
    assert settings.openrouter_api_key == "test-key"
    assert settings.chroma_collection_name == "rag_docs"
    assert settings.max_upload_mb == 25
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.min_relevance_score == 0.75
    assert settings.app_log_level == "INFO"


def test_load_environment_from_dotenv_sets_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_environment_from_dotenv(str(dotenv_path))

    assert loaded is True
    assert os.getenv("OPENROUTER_API_KEY") == "from-dotenv"
