import asyncio
from types import MappingProxyType

import pytest

from app.services.azure_openai_chat_provider import AzureOpenAIChatProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def _build_provider() -> AzureOpenAIChatProvider:
    return AzureOpenAIChatProvider(
        endpoint="https://azure-openai.example.com",
        api_key="azure-key",
        api_version="2024-10-21",
        deployments=MappingProxyType({"gpt-4o-mini": "chat-gpt4o-mini"}),
    )


def test_generate_chat_uses_deployment_url_and_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
    ) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": "hi"}}]},
        )

    monkeypatch.setattr(AzureOpenAIChatProvider, "_post_json", fake_post_json)
    provider = _build_provider()

    response = asyncio.run(
        provider.generate_chat_response_with_model(
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert response == "hi"
    assert captured["url"] == (
        "https://azure-openai.example.com/openai/deployments/chat-gpt4o-mini/"
        "chat/completions?api-version=2024-10-21"
    )
    assert captured["headers"] == {
        "api-key": "azure-key",
        "Content-Type": "application/json",
    }
    assert captured["payload"] == {
        "stream": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }


def test_stream_chat_parsing_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream_post_data_lines(self, url: str, headers: dict, payload: dict):
        yield '{"choices":[{"delta":{"content":"One "}}]}'
        yield '{"choices":[{"delta":{"content":"two"}}]}'
        yield '{"choices":[{"delta":{}}]}'

    monkeypatch.setattr(
        AzureOpenAIChatProvider,
        "_stream_post_data_lines",
        fake_stream_post_data_lines,
    )
    provider = _build_provider()

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in provider.stream_chat_response_with_model(
                model="gpt-4o-mini",
                system_prompt="system",
                user_prompt="user",
            )
        ]

    chunks = asyncio.run(collect())

    assert chunks == ["One ", "two"]


def test_generate_chat_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
    ) -> FakeResponse:
        return FakeResponse(status_code=429, payload={"error": "busy"}, text="busy")

    monkeypatch.setattr(AzureOpenAIChatProvider, "_post_json", fake_post_json)
    provider = _build_provider()

    with pytest.raises(RuntimeError, match="Azure OpenAI chat request failed"):
        asyncio.run(
            provider.generate_chat_response_with_model(
                model="gpt-4o-mini",
                system_prompt="system",
                user_prompt="user",
            )
        )
