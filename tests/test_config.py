import os

import pytest

from app.config import Settings, load_environment_from_dotenv


def test_missing_required_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
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

    with pytest.raises(
        ValueError, match="Missing required environment variable: OPENROUTER_API_KEY"
    ):
        Settings.from_env()


def test_invalid_integer_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openrouter/test-chat")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv(
        "OPENROUTER_BATTLEGROUND_MODELS",
        "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet",
    )
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "bad-int")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.4")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")

    with pytest.raises(
        ValueError, match="Invalid integer for environment variable MAX_UPLOAD_MB"
    ):
        Settings.from_env()


def test_settings_from_env_success(required_env: None) -> None:
    settings = Settings.from_env()
    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_battleground_models == (
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    )
    assert settings.chroma_collection_name == "rag_docs"
    assert settings.max_upload_mb == 25
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.min_relevance_score == 0.4
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


def test_missing_battleground_models_raises(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_BATTLEGROUND_MODELS", raising=False)

    with pytest.raises(
        ValueError,
        match="Missing required environment variable: OPENROUTER_BATTLEGROUND_MODELS",
    ):
        Settings.from_env()


def test_invalid_battleground_models_empty_entry_raises(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_BATTLEGROUND_MODELS", "openai/gpt-4o,,anthropic/claude"
    )

    with pytest.raises(
        ValueError,
        match="OPENROUTER_BATTLEGROUND_MODELS contains empty model id",
    ):
        Settings.from_env()


def test_battleground_models_trimmed_from_csv(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_BATTLEGROUND_MODELS",
        " openai/gpt-4o-mini , anthropic/claude-3.5-sonnet ",
    )

    settings = Settings.from_env()

    assert settings.openrouter_battleground_models == (
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    )


def test_battleground_models_require_at_least_two_distinct_ids(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_BATTLEGROUND_MODELS", "openai/gpt-4o-mini")

    with pytest.raises(
        ValueError,
        match=(
            "OPENROUTER_BATTLEGROUND_MODELS must contain at least 2 distinct model ids"
        ),
    ):
        Settings.from_env()


def test_battleground_models_reject_duplicate_model_ids(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_BATTLEGROUND_MODELS",
        "openai/gpt-4o-mini,openai/gpt-4o-mini,anthropic/claude-3.5-sonnet",
    )

    with pytest.raises(
        ValueError,
        match=(
            "OPENROUTER_BATTLEGROUND_MODELS must not contain duplicate model ids: "
            "openai/gpt-4o-mini"
        ),
    ):
        Settings.from_env()


def test_battleground_models_are_immutable(required_env: None) -> None:
    settings = Settings.from_env()

    assert isinstance(settings.openrouter_battleground_models, tuple)
    with pytest.raises(AttributeError):
        settings.openrouter_battleground_models.append("meta/llama-3.1-8b-instruct")
