import asyncio
import pytest

from app.services.openrouter_client import OpenRouterClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
        if url == "":
            raise AssertionError("url must not be empty")
        if "Authorization" not in headers:
            raise AssertionError("Authorization header must be present")
        if len(json) == 0:
            raise AssertionError("json payload must not be empty")
        return self._response


def test_embed_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(status_code=401, payload={"error": "bad key"}, text="bad key")
    async def fake_post_json(self, path: str, payload: dict) -> FakeResponse:
        client = FakeAsyncClient(response)
        return await client.post(path, {"Authorization": "Bearer k"}, payload)

    monkeypatch.setattr(OpenRouterClient, "_post_json", fake_post_json)
    client = OpenRouterClient(
        api_key="k", embed_model="openrouter/embed", chat_model="openrouter/chat"
    )

    with pytest.raises(RuntimeError, match="OpenRouter embeddings request failed"):
        asyncio.run(client.embed_texts(["hello"]))


def test_embed_returns_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        status_code=200,
        payload={"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
    )
    async def fake_post_json(self, path: str, payload: dict) -> FakeResponse:
        client = FakeAsyncClient(response)
        return await client.post(path, {"Authorization": "Bearer k"}, payload)

    monkeypatch.setattr(OpenRouterClient, "_post_json", fake_post_json)
    client = OpenRouterClient(
        api_key="k", embed_model="openrouter/embed", chat_model="openrouter/chat"
    )

    vectors = asyncio.run(client.embed_texts(["hello", "world"]))
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_chat_raises_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(status_code=500, payload={"error": "boom"}, text="boom")
    async def fake_post_json(self, path: str, payload: dict) -> FakeResponse:
        client = FakeAsyncClient(response)
        return await client.post(path, {"Authorization": "Bearer k"}, payload)

    monkeypatch.setattr(OpenRouterClient, "_post_json", fake_post_json)
    client = OpenRouterClient(
        api_key="k", embed_model="openrouter/embed", chat_model="openrouter/chat"
    )

    with pytest.raises(RuntimeError, match="OpenRouter chat request failed"):
        asyncio.run(client.generate_chat_response("system", "user"))


def test_chat_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        status_code=200,
        payload={"choices": [{"message": {"content": "grounded answer"}}]},
    )
    async def fake_post_json(self, path: str, payload: dict) -> FakeResponse:
        client = FakeAsyncClient(response)
        return await client.post(path, {"Authorization": "Bearer k"}, payload)

    monkeypatch.setattr(OpenRouterClient, "_post_json", fake_post_json)
    client = OpenRouterClient(
        api_key="k", embed_model="openrouter/embed", chat_model="openrouter/chat"
    )

    result = asyncio.run(client.generate_chat_response("system", "user"))
    assert result == "grounded answer"
