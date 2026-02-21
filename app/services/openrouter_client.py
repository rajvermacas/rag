"""OpenRouter API client for embeddings and chat completions."""

import logging
from typing import Any


logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Client wrapper for OpenRouter endpoints."""

    def __init__(self, api_key: str, embed_model: str, chat_model: str) -> None:
        if api_key.strip() == "":
            raise ValueError("api_key must not be empty")
        if embed_model.strip() == "":
            raise ValueError("embed_model must not be empty")
        if chat_model.strip() == "":
            raise ValueError("chat_model must not be empty")
        self._api_key = api_key
        self._embed_model = embed_model
        self._chat_model = chat_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 0:
            raise ValueError("texts must not be empty")

        logger.info("openrouter_embed_started text_count=%s", len(texts))
        payload = {"model": self._embed_model, "input": texts}
        response = await self._post_json("/embeddings", payload)
        if response.status_code != 200:
            raise RuntimeError(
                "OpenRouter embeddings request failed: "
                f"{response.status_code} {response.text}"
            )

        data = response.json().get("data")
        if not isinstance(data, list):
            raise ValueError("OpenRouter embeddings response missing data list")
        embeddings = [item["embedding"] for item in data]
        logger.info("openrouter_embed_completed embedding_count=%s", len(embeddings))
        return embeddings

    async def generate_chat_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        if system_prompt.strip() == "":
            raise ValueError("system_prompt must not be empty")
        if user_prompt.strip() == "":
            raise ValueError("user_prompt must not be empty")

        logger.info("openrouter_chat_started")
        payload = {
            "model": self._chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = await self._post_json("/chat/completions", payload)
        if response.status_code != 200:
            raise RuntimeError(
                "OpenRouter chat request failed: "
                f"{response.status_code} {response.text}"
            )

        choices = response.json().get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError("OpenRouter chat response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("OpenRouter chat response missing message content")
        content = message["content"]
        logger.info("openrouter_chat_completed response_length=%s", len(content))
        return content

    async def _post_json(self, path: str, payload: dict) -> Any:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency for OpenRouter client: httpx") from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(
                f"{OPENROUTER_BASE_URL}{path}",
                headers=headers,
                json=payload,
            )
