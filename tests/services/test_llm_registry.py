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


def test_registry_builds_openrouter_llm_profile() -> None:
    registry = LLMRegistry(settings_with_openrouter_profile())

    llm = registry.get_llm("openrouter_lab", "openai/gpt-4o-mini")

    assert llm is not None


def test_registry_rejects_unknown_backend_id() -> None:
    registry = LLMRegistry(settings_with_openrouter_profile())

    with pytest.raises(ValueError, match="backend_id is not allowed"):
        registry.get_llm("missing", "openai/gpt-4o-mini")
