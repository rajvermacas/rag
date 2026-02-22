"""LLM registry for backend/model constrained LlamaIndex provider instances."""

import logging
from types import MappingProxyType
from typing import Any, Mapping

from app.constants import (
    OPENROUTER_API_BASE_URL,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
)
from app.config import ChatBackendProfile
from app.services.chat_provider_models import require_non_empty


logger = logging.getLogger(__name__)


class LLMRegistry:
    """Builds provider-specific LlamaIndex LLMs from validated backend profiles."""

    def __init__(self, backend_profiles: Mapping[str, ChatBackendProfile]) -> None:
        if len(backend_profiles) == 0:
            raise ValueError("backend_profiles must not be empty")
        self._profiles = MappingProxyType(dict(backend_profiles))

    def get_llm(self, backend_id: str, model: str) -> Any:
        profile = self._resolve_profile(backend_id)
        normalized_model = self._validate_model(profile, model)
        logger.info(
            "llm_registry_build_llm backend_id=%s provider=%s model=%s",
            profile.backend_id,
            profile.provider,
            normalized_model,
        )
        return _build_llamaindex_llm(profile, normalized_model)

    def _resolve_profile(self, backend_id: str) -> ChatBackendProfile:
        normalized_backend_id = require_non_empty(backend_id, "backend_id")
        if normalized_backend_id not in self._profiles:
            raise ValueError("backend_id is not allowed")
        return self._profiles[normalized_backend_id]

    def _validate_model(self, profile: ChatBackendProfile, model: str) -> str:
        normalized_model = require_non_empty(model, "model")
        if normalized_model not in profile.models:
            raise ValueError("model is not allowed for backend_id")
        return normalized_model


def _build_llamaindex_llm(profile: ChatBackendProfile, model: str) -> Any:
    if profile.provider == PROVIDER_OPENROUTER:
        return _build_openrouter_llm(profile, model)
    if profile.provider == PROVIDER_OPENAI:
        return _build_openai_llm(profile, model)
    if profile.provider == PROVIDER_AZURE_OPENAI:
        return _build_azure_openai_llm(profile, model)
    raise ValueError(f"unsupported provider for backend profile: {profile.provider}")


def _build_openrouter_llm(profile: ChatBackendProfile, model: str) -> Any:
    openai_like = _import_openai_like_class()
    logger.info(
        "llm_registry_openrouter_configured backend_id=%s model=%s api_base=%s "
        "is_chat_model=%s",
        profile.backend_id,
        model,
        OPENROUTER_API_BASE_URL,
        True,
    )
    return openai_like(
        model=model,
        api_key=profile.api_key,
        api_base=OPENROUTER_API_BASE_URL,
        is_chat_model=True,
    )


def _build_openai_llm(profile: ChatBackendProfile, model: str) -> Any:
    openai_llm = _import_openai_class()
    logger.info(
        "llm_registry_openai_configured backend_id=%s model=%s",
        profile.backend_id,
        model,
    )
    return openai_llm(
        model=model,
        api_key=profile.api_key,
    )


def _build_azure_openai_llm(profile: ChatBackendProfile, model: str) -> Any:
    azure_openai_llm = _import_azure_openai_class()
    endpoint = _require_profile_value(profile.azure_endpoint, "azure_endpoint")
    api_version = _require_profile_value(profile.azure_api_version, "azure_api_version")
    deployment = _resolve_azure_deployment(profile, model)
    logger.info(
        "llm_registry_azure_openai_configured backend_id=%s model=%s deployment=%s",
        profile.backend_id,
        model,
        deployment,
    )
    return azure_openai_llm(
        model=model,
        api_key=profile.api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
    )


def _require_profile_value(value: str | None, field_name: str) -> str:
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} is required for azure_openai backend")
    return value


def _resolve_azure_deployment(profile: ChatBackendProfile, model: str) -> str:
    if model not in profile.azure_deployments:
        raise ValueError(f"azure_deployments missing mapping for model: {model}")
    deployment = profile.azure_deployments[model]
    if deployment.strip() == "":
        raise ValueError(f"azure_deployments contains empty deployment for model: {model}")
    return deployment


def _import_openai_like_class() -> Any:
    try:
        from llama_index.llms.openai_like import OpenAILike
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency for openrouter provider: "
            "install llama-index-llms-openai-like"
        ) from exc
    return OpenAILike


def _import_openai_class() -> Any:
    try:
        from llama_index.llms.openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency for openai provider: install llama-index-llms-openai"
        ) from exc
    return OpenAI


def _import_azure_openai_class() -> Any:
    try:
        from llama_index.llms.azure_openai import AzureOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency for azure_openai provider: "
            "install llama-index-llms-azure-openai"
        ) from exc
    return AzureOpenAI
