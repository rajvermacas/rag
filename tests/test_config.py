import os

import pytest

from app.config import Settings, load_environment_from_dotenv


def _set_common_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_EMBED_MODEL", "openrouter/test-embed")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma-test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "rag_docs")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.4")
    monkeypatch.setenv("APP_LOG_LEVEL", "INFO")


def _set_openrouter_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_id: str,
    models: str,
) -> None:
    backend_token = backend_id.upper()
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_PROVIDER", "openrouter")
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_MODELS", models)
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_API_KEY", "test-openrouter-chat-key")


def _set_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_id: str,
    models: str,
) -> None:
    backend_token = backend_id.upper()
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_PROVIDER", "openai")
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_MODELS", models)
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_API_KEY", "test-openai-chat-key")


def _set_azure_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_id: str,
    models: str,
    deployments: str,
) -> None:
    backend_token = backend_id.upper()
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_PROVIDER", "azure_openai")
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_MODELS", models)
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_API_KEY", "test-azure-key")
    monkeypatch.setenv(
        f"CHAT_BACKEND_{backend_token}_AZURE_ENDPOINT",
        "https://azure-openai.example.com",
    )
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv(f"CHAT_BACKEND_{backend_token}_AZURE_DEPLOYMENTS", deployments)


def test_missing_openrouter_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="lab_vllm",
        models="openai/gpt-4o-mini",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="Missing required environment variable: OPENROUTER_API_KEY",
    ):
        Settings.from_env()


def test_missing_chat_backend_ids_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Missing required environment variable: CHAT_BACKEND_IDS",
    ):
        Settings.from_env()


def test_unknown_backend_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_PROVIDER", "unexpected_provider")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_MODELS", "gpt-4o-mini")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_API_KEY", "test-chat-key")

    with pytest.raises(
        ValueError,
        match=(
            "CHAT_BACKEND_LAB_VLLM_PROVIDER must be one of: "
            "openrouter, openai, azure_openai"
        ),
    ):
        Settings.from_env()


def test_settings_accepts_openrouter_openai_and_azure_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "openrouter_lab,openai_prod,azure_prod")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="openrouter_lab",
        models="openai/gpt-4o-mini",
    )
    _set_openai_backend(
        monkeypatch=monkeypatch,
        backend_id="openai_prod",
        models="gpt-4o-mini",
    )
    _set_azure_backend(
        monkeypatch=monkeypatch,
        backend_id="azure_prod",
        models="gpt-4o-mini",
        deployments="gpt-4o-mini=chat-gpt4o-mini",
    )

    settings = Settings.from_env()

    assert settings.chat_backend_profiles["openrouter_lab"].provider == "openrouter"
    assert settings.chat_backend_profiles["openai_prod"].provider == "openai"
    assert settings.chat_backend_profiles["azure_prod"].provider == "azure_openai"


def test_openai_backend_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_PROVIDER", "openai")
    monkeypatch.setenv("CHAT_BACKEND_LAB_VLLM_MODELS", "gpt-4o-mini")
    monkeypatch.delenv("CHAT_BACKEND_LAB_VLLM_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="Missing required environment variable: CHAT_BACKEND_LAB_VLLM_API_KEY",
    ):
        Settings.from_env()


def test_backend_models_reject_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="lab_vllm",
        models="openai/gpt-4o-mini,openai/gpt-4o-mini",
    )

    with pytest.raises(
        ValueError,
        match=(
            "CHAT_BACKEND_LAB_VLLM_MODELS must not contain duplicate model ids: "
            "openai/gpt-4o-mini"
        ),
    ):
        Settings.from_env()


def test_azure_backend_requires_deployments_for_all_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "azure_prod")
    _set_azure_backend(
        monkeypatch=monkeypatch,
        backend_id="azure_prod",
        models="gpt-4o-mini,gpt-4.1-mini",
        deployments="gpt-4o-mini=chat-gpt4o-mini",
    )

    with pytest.raises(
        ValueError,
        match=(
            "CHAT_BACKEND_AZURE_PROD_AZURE_DEPLOYMENTS missing model mappings for: "
            "gpt-4.1-mini"
        ),
    ):
        Settings.from_env()


def test_settings_parse_two_chat_backends_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm,azure_prod")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="lab_vllm",
        models="openai/gpt-4o-mini,anthropic/claude-3.5-sonnet",
    )
    _set_azure_backend(
        monkeypatch=monkeypatch,
        backend_id="azure_prod",
        models="gpt-4o-mini",
        deployments="gpt-4o-mini=chat-gpt4o-mini",
    )

    settings = Settings.from_env()

    assert settings.chat_backend_profiles["lab_vllm"].provider == "openrouter"
    assert (
        settings.chat_backend_profiles["azure_prod"].azure_deployments["gpt-4o-mini"]
        == "chat-gpt4o-mini"
    )


def test_invalid_integer_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="lab_vllm",
        models="openai/gpt-4o-mini",
    )
    monkeypatch.setenv("MAX_UPLOAD_MB", "bad-int")

    with pytest.raises(
        ValueError,
        match="Invalid integer for environment variable MAX_UPLOAD_MB",
    ):
        Settings.from_env()


def test_settings_from_env_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_common_env(monkeypatch)
    monkeypatch.setenv("CHAT_BACKEND_IDS", "lab_vllm")
    _set_openrouter_backend(
        monkeypatch=monkeypatch,
        backend_id="lab_vllm",
        models="openai/gpt-4o-mini",
    )

    settings = Settings.from_env()

    assert settings.openrouter_api_key == "test-openrouter-key"
    assert settings.chroma_collection_name == "rag_docs"
    assert settings.max_upload_mb == 25
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.min_relevance_score == 0.4
    assert settings.app_log_level == "INFO"


def test_load_environment_from_dotenv_sets_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_environment_from_dotenv(str(dotenv_path))

    assert loaded is True
    assert os.getenv("OPENROUTER_API_KEY") == "from-dotenv"
