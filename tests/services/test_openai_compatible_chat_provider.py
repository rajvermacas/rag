import asyncio

import pytest

from app.services.openai_compatible_chat_provider import OpenAICompatibleChatProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_generate_chat_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
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
            payload={"choices": [{"message": {"content": "hello"}}]},
        )

    monkeypatch.setattr(OpenAICompatibleChatProvider, "_post_json", fake_post_json)
    provider = OpenAICompatibleChatProvider(
        base_url="https://lab.example.com/v1",
        api_key="chat-key",
    )

    response = asyncio.run(
        provider.generate_chat_response_with_model(
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert response == "hello"
    assert captured["url"] == "https://lab.example.com/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer chat-key",
        "Content-Type": "application/json",
    }
    assert captured["payload"] == {
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }


def test_stream_chat_parsing_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream_post_data_lines(self, url: str, headers: dict, payload: dict):
        yield '{"choices":[{"delta":{"content":"Hello "}}]}'
        yield '{"choices":[{"delta":{"content":"world"}}]}'
        yield '{"choices":[{"delta":{}}]}'

    monkeypatch.setattr(
        OpenAICompatibleChatProvider,
        "_stream_post_data_lines",
        fake_stream_post_data_lines,
    )
    provider = OpenAICompatibleChatProvider(
        base_url="https://lab.example.com/v1",
        api_key="chat-key",
    )

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

    assert chunks == ["Hello ", "world"]


def test_generate_chat_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
    ) -> FakeResponse:
        return FakeResponse(status_code=503, payload={"error": "down"}, text="down")

    monkeypatch.setattr(OpenAICompatibleChatProvider, "_post_json", fake_post_json)
    provider = OpenAICompatibleChatProvider(
        base_url="https://lab.example.com/v1",
        api_key="chat-key",
    )

    with pytest.raises(RuntimeError, match="OpenAI-compatible chat request failed"):
        asyncio.run(
            provider.generate_chat_response_with_model(
                model="gpt-4o-mini",
                system_prompt="system",
                user_prompt="user",
            )
        )
