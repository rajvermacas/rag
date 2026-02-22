import asyncio
from types import MappingProxyType
from typing import AsyncIterator

import pytest

from app.config import ChatBackendProfile
from app.services.chat_provider_router import ChatProviderRouter


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def generate_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((model, system_prompt, user_prompt))
        return "ok"

    async def stream_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        self.calls.append((model, system_prompt, user_prompt))
        yield "chunk"


def _build_profiles() -> MappingProxyType:
    return MappingProxyType(
        {
            "lab_vllm": ChatBackendProfile(
                backend_id="lab_vllm",
                provider="openai_compatible",
                models=("gpt-4o-mini",),
                api_key="key",
                base_url="https://lab.example.com/v1",
                azure_endpoint=None,
                azure_api_version=None,
                azure_deployments=MappingProxyType({}),
            )
        }
    )


def test_router_rejects_unknown_backend_id() -> None:
    router = ChatProviderRouter(
        backend_profiles=_build_profiles(),
        providers={"lab_vllm": FakeProvider()},
    )

    with pytest.raises(ValueError, match="backend_id is not allowed"):
        asyncio.run(
            router.generate_chat_response_with_backend(
                backend_id="missing",
                model="gpt-4o-mini",
                system_prompt="s",
                user_prompt="u",
            )
        )


def test_router_rejects_model_not_in_backend_allowlist() -> None:
    router = ChatProviderRouter(
        backend_profiles=_build_profiles(),
        providers={"lab_vllm": FakeProvider()},
    )

    with pytest.raises(ValueError, match="model is not allowed for backend_id"):
        asyncio.run(
            router.generate_chat_response_with_backend(
                backend_id="lab_vllm",
                model="gpt-4.1-mini",
                system_prompt="s",
                user_prompt="u",
            )
        )


def test_router_delegates_to_selected_backend_provider() -> None:
    provider = FakeProvider()
    router = ChatProviderRouter(
        backend_profiles=_build_profiles(),
        providers={"lab_vllm": provider},
    )

    response = asyncio.run(
        router.generate_chat_response_with_backend(
            backend_id="lab_vllm",
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert response == "ok"
    assert provider.calls == [("gpt-4o-mini", "system", "user")]
