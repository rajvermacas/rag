"""Configuration loading with strict required environment variables."""

from dataclasses import dataclass
import logging
from types import MappingProxyType
import os
import re
from typing import Mapping

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
_PROVIDER_OPENROUTER = "openrouter"
_PROVIDER_OPENAI = "openai"
_PROVIDER_AZURE_OPENAI = "azure_openai"
_ALLOWED_PROVIDERS = (_PROVIDER_OPENROUTER, _PROVIDER_OPENAI, _PROVIDER_AZURE_OPENAI)
_BACKEND_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class ChatBackendProfile:
    """Immutable chat backend profile definition."""

    backend_id: str
    provider: str
    models: tuple[str, ...]
    api_key: str
    base_url: str | None
    azure_endpoint: str | None
    azure_api_version: str | None
    azure_deployments: Mapping[str, str]


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_embed_model: str
    chat_backend_profiles: Mapping[str, ChatBackendProfile]
    chroma_persist_dir: str
    chroma_collection_name: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    min_relevance_score: float
    app_log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        backend_ids = _parse_required_csv_with_uniqueness(
            "CHAT_BACKEND_IDS",
            duplicate_error_label="backend ids",
        )
        chat_backend_profiles = _parse_chat_backend_profiles(backend_ids)
        return cls(
            openrouter_api_key=_require_env("OPENROUTER_API_KEY"),
            openrouter_embed_model=_require_env("OPENROUTER_EMBED_MODEL"),
            chat_backend_profiles=chat_backend_profiles,
            chroma_persist_dir=_require_env("CHROMA_PERSIST_DIR"),
            chroma_collection_name=_require_env("CHROMA_COLLECTION_NAME"),
            max_upload_mb=_parse_int("MAX_UPLOAD_MB"),
            chunk_size=_parse_int("CHUNK_SIZE"),
            chunk_overlap=_parse_int("CHUNK_OVERLAP"),
            retrieval_top_k=_parse_int("RETRIEVAL_TOP_K"),
            min_relevance_score=_parse_float("MIN_RELEVANCE_SCORE"),
            app_log_level=_require_env("APP_LOG_LEVEL"),
        )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_int(name: str) -> int:
    raw_value = _require_env(name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for environment variable {name}: {raw_value}"
        ) from exc


def _parse_float(name: str) -> float:
    raw_value = _require_env(name)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float for environment variable {name}: {raw_value}"
        ) from exc


def _parse_required_csv_with_uniqueness(
    name: str,
    duplicate_error_label: str,
) -> tuple[str, ...]:
    raw_value = _require_env(name)
    values = tuple(value.strip() for value in raw_value.split(","))
    if len(values) == 0 or any(value == "" for value in values):
        raise ValueError(f"{name} contains empty value")
    duplicate_values = _find_duplicate_values(values)
    if len(duplicate_values) > 0:
        raise ValueError(
            f"{name} must not contain duplicate {duplicate_error_label}: "
            f"{', '.join(duplicate_values)}"
        )
    logger.debug("parsed_required_csv name=%s value_count=%d", name, len(values))
    return values


def _find_duplicate_values(values: tuple[str, ...]) -> tuple[str, ...]:
    duplicate_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen and value not in duplicate_values:
            duplicate_values.append(value)
        seen.add(value)
    return tuple(duplicate_values)


def _parse_chat_backend_profiles(
    backend_ids: tuple[str, ...],
) -> Mapping[str, ChatBackendProfile]:
    profiles: dict[str, ChatBackendProfile] = {}
    for backend_id in backend_ids:
        if not _BACKEND_ID_PATTERN.match(backend_id):
            raise ValueError(
                f"CHAT_BACKEND_IDS contains invalid backend id: {backend_id}"
            )
        profile = _parse_chat_backend_profile(backend_id)
        profiles[backend_id] = profile
    logger.info("validated_chat_backend_profiles backend_count=%s", len(profiles))
    return MappingProxyType(profiles)


def _parse_chat_backend_profile(backend_id: str) -> ChatBackendProfile:
    backend_token = backend_id.upper()
    provider_env = f"CHAT_BACKEND_{backend_token}_PROVIDER"
    provider = _require_env(provider_env)
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(
            f"{provider_env} must be one of: "
            f"{_PROVIDER_OPENROUTER}, {_PROVIDER_OPENAI}, {_PROVIDER_AZURE_OPENAI}"
        )
    models = _parse_required_csv_with_uniqueness(
        f"CHAT_BACKEND_{backend_token}_MODELS",
        duplicate_error_label="model ids",
    )
    api_key = _require_env(f"CHAT_BACKEND_{backend_token}_API_KEY")
    if provider == _PROVIDER_OPENROUTER:
        return _build_openrouter_profile(backend_id, models, api_key)
    if provider == _PROVIDER_OPENAI:
        return _build_openai_profile(backend_id, models, api_key)
    return _build_azure_profile(backend_id, backend_token, models, api_key)


def _build_openrouter_profile(
    backend_id: str,
    models: tuple[str, ...],
    api_key: str,
) -> ChatBackendProfile:
    logger.info(
        "validated_chat_backend backend_id=%s provider=%s model_count=%s",
        backend_id,
        _PROVIDER_OPENROUTER,
        len(models),
    )
    return ChatBackendProfile(
        backend_id=backend_id,
        provider=_PROVIDER_OPENROUTER,
        models=models,
        api_key=api_key,
        base_url=None,
        azure_endpoint=None,
        azure_api_version=None,
        azure_deployments=MappingProxyType({}),
    )


def _build_openai_profile(
    backend_id: str,
    models: tuple[str, ...],
    api_key: str,
) -> ChatBackendProfile:
    logger.info(
        "validated_chat_backend backend_id=%s provider=%s model_count=%s",
        backend_id,
        _PROVIDER_OPENAI,
        len(models),
    )
    return ChatBackendProfile(
        backend_id=backend_id,
        provider=_PROVIDER_OPENAI,
        models=models,
        api_key=api_key,
        base_url=None,
        azure_endpoint=None,
        azure_api_version=None,
        azure_deployments=MappingProxyType({}),
    )


def _build_azure_profile(
    backend_id: str,
    backend_token: str,
    models: tuple[str, ...],
    api_key: str,
) -> ChatBackendProfile:
    endpoint = _require_env(f"CHAT_BACKEND_{backend_token}_AZURE_ENDPOINT")
    api_version = _require_env(f"CHAT_BACKEND_{backend_token}_AZURE_API_VERSION")
    deployments_env = f"CHAT_BACKEND_{backend_token}_AZURE_DEPLOYMENTS"
    deployments = _parse_azure_deployments(deployments_env)
    _validate_azure_deployment_coverage(deployments_env, models, deployments)
    logger.info(
        "validated_chat_backend backend_id=%s provider=%s model_count=%s",
        backend_id,
        _PROVIDER_AZURE_OPENAI,
        len(models),
    )
    return ChatBackendProfile(
        backend_id=backend_id,
        provider=_PROVIDER_AZURE_OPENAI,
        models=models,
        api_key=api_key,
        base_url=None,
        azure_endpoint=endpoint,
        azure_api_version=api_version,
        azure_deployments=deployments,
    )


def _parse_azure_deployments(name: str) -> Mapping[str, str]:
    entries = _parse_required_csv_with_uniqueness(name, duplicate_error_label="entries")
    deployments: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{name} must use model=deployment mapping entries")
        model, deployment = entry.split("=", maxsplit=1)
        model_id = model.strip()
        deployment_name = deployment.strip()
        if model_id == "" or deployment_name == "":
            raise ValueError(f"{name} contains empty model or deployment value")
        if model_id in deployments:
            raise ValueError(f"{name} contains duplicate model mapping: {model_id}")
        deployments[model_id] = deployment_name
    return MappingProxyType(deployments)


def _validate_azure_deployment_coverage(
    deployments_env: str,
    models: tuple[str, ...],
    deployments: Mapping[str, str],
) -> None:
    missing_models = tuple(model for model in models if model not in deployments)
    if len(missing_models) > 0:
        raise ValueError(
            f"{deployments_env} missing model mappings for: {', '.join(missing_models)}"
        )


def load_environment_from_dotenv(dotenv_path: str) -> bool:
    if dotenv_path.strip() == "":
        raise ValueError("dotenv_path must not be empty")
    loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
    logger.info("dotenv_load_attempted dotenv_path=%s loaded=%s", dotenv_path, loaded)
    return loaded
