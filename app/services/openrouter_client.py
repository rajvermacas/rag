"""OpenRouter API client for embeddings and chat completions."""

import json
import logging
from typing import Any, AsyncIterator


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
        _validate_prompt_inputs(system_prompt, user_prompt)

        logger.info("openrouter_chat_started")
        payload = _build_chat_payload(self._chat_model, system_prompt, user_prompt, stream=False)
        response = await self._post_json("/chat/completions", payload)
        if response.status_code != 200:
            raise RuntimeError(
                "OpenRouter chat request failed: "
                f"{response.status_code} {response.text}"
            )
        content = _extract_chat_content(response.json())
        logger.info("openrouter_chat_completed response_length=%s", len(content))
        return content

    async def stream_chat_response(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        _validate_prompt_inputs(system_prompt, user_prompt)
        logger.info("openrouter_chat_stream_started")
        payload = _build_chat_payload(self._chat_model, system_prompt, user_prompt, stream=True)
        async for data_line in self._stream_post_data_lines("/chat/completions", payload):
            chunk = _extract_stream_chunk(data_line)
            if chunk != "":
                yield chunk
        logger.info("openrouter_chat_stream_completed")

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

    async def _stream_post_data_lines(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency for OpenRouter client: httpx") from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE_URL}{path}",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode("utf-8")
                    raise RuntimeError(
                        "OpenRouter chat stream request failed: "
                        f"{response.status_code} {error_body}"
                    )
                async for raw_line in response.aiter_lines():
                    if raw_line == "":
                        continue
                    if not raw_line.startswith("data: "):
                        continue
                    payload_line = raw_line[6:]
                    if payload_line == "[DONE]":
                        return
                    yield payload_line


def _validate_prompt_inputs(system_prompt: str, user_prompt: str) -> None:
    if system_prompt.strip() == "":
        raise ValueError("system_prompt must not be empty")
    if user_prompt.strip() == "":
        raise ValueError("user_prompt must not be empty")


def _build_chat_payload(
    chat_model: str, system_prompt: str, user_prompt: str, stream: bool
) -> dict[str, Any]:
    return {
        "model": chat_model,
        "stream": stream,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _extract_chat_content(payload: dict[str, Any]) -> str:
    if "choices" not in payload:
        raise ValueError("OpenRouter chat response missing choices")
    choices = payload["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("OpenRouter chat response missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict) or "message" not in first_choice:
        raise ValueError("OpenRouter chat response missing message")
    message = first_choice["message"]
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError("OpenRouter chat response missing message content")
    content = message["content"]
    if not isinstance(content, str):
        raise ValueError("OpenRouter chat response content must be a string")
    return content


def _extract_stream_chunk(data_line: str) -> str:
    payload = json.loads(data_line)
    if "choices" not in payload:
        raise ValueError("OpenRouter chat stream event missing choices")
    choices = payload["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("OpenRouter chat stream event missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenRouter chat stream event choice must be an object")
    if "delta" not in first_choice:
        return ""
    delta = first_choice["delta"]
    if not isinstance(delta, dict):
        raise ValueError("OpenRouter chat stream delta must be an object")
    if "content" not in delta:
        return ""
    content = delta["content"]
    if not isinstance(content, str):
        raise ValueError("OpenRouter chat stream content must be a string")
    return content
