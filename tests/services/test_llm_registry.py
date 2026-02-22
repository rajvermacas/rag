from types import MappingProxyType

import pytest

from app.config import ChatBackendProfile
from app.services.llm_registry import LLMRegistry


def settings_with_openrouter_profile() -> MappingProxyType[str, ChatBackendProfile]:
    return MappingProxyType(
        {
            "openrouter_lab": ChatBackendProfile(
                backend_id="openrouter_lab",
                provider="openrouter",
                models=("openai/gpt-4o-mini",),
                api_key="test-openrouter-key",
                base_url=None,
                azure_endpoint=None,
                azure_api_version=None,
                azure_deployments=MappingProxyType({}),
            )
        }
    )


def settings_with_azure_profile() -> MappingProxyType[str, ChatBackendProfile]:
    return MappingProxyType(
        {
            "azure_prod": ChatBackendProfile(
                backend_id="azure_prod",
                provider="azure_openai",
                models=("gpt-4o-mini",),
                api_key="test-azure-key",
                base_url=None,
                azure_endpoint="https://azure-openai.example.com",
                azure_api_version="2024-10-21",
                azure_deployments=MappingProxyType({"gpt-4o-mini": "chat-gpt4o-mini"}),
            )
        }
    )


def test_registry_builds_openrouter_llm_profile() -> None:
    registry = LLMRegistry(settings_with_openrouter_profile())

    llm = registry.get_llm("openrouter_lab", "openai/gpt-4o-mini")

    assert llm is not None


def test_registry_rejects_unknown_backend_id() -> None:
    registry = LLMRegistry(settings_with_openrouter_profile())

    with pytest.raises(ValueError, match="backend_id is not allowed"):
        registry.get_llm("missing", "openai/gpt-4o-mini")


def test_llm_registry_openrouter_configuration_uses_openai_compatible_base_url() -> None:
    registry = LLMRegistry(settings_with_openrouter_profile())

    llm = registry.get_llm("openrouter_lab", "openai/gpt-4o-mini")

    assert getattr(llm, "api_base") == "https://openrouter.ai/api/v1"
    assert getattr(llm, "model") == "openai/gpt-4o-mini"
    assert getattr(llm, "is_chat_model") is True


def test_registry_azure_provider_requires_model_deployment_mapping() -> None:
    invalid_profiles = MappingProxyType(
        {
            "azure_prod": ChatBackendProfile(
                backend_id="azure_prod",
                provider="azure_openai",
                models=("gpt-4o-mini",),
                api_key="test-azure-key",
                base_url=None,
                azure_endpoint="https://azure-openai.example.com",
                azure_api_version="2024-10-21",
                azure_deployments=MappingProxyType({}),
            )
        }
    )
    registry = LLMRegistry(invalid_profiles)

    with pytest.raises(ValueError, match="azure_deployments missing mapping for model"):
        registry.get_llm("azure_prod", "gpt-4o-mini")


def test_registry_builds_azure_openai_llm_profile() -> None:
    registry = LLMRegistry(settings_with_azure_profile())

    llm = registry.get_llm("azure_prod", "gpt-4o-mini")

    assert llm is not None
    assert getattr(llm, "azure_endpoint") == "https://azure-openai.example.com"
