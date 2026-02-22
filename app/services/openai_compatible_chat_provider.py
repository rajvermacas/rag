"""Provider client for OpenAI-compatible chat completion APIs."""

import logging
from typing import Any, AsyncIterator

from app.services.chat_provider_models import (
    build_chat_payload,
    extract_chat_content,
    extract_stream_chunk,
    require_non_empty,
    validate_prompt_inputs,
)


logger = logging.getLogger(__name__)


class OpenAICompatibleChatProvider:
    """Chat provider for arbitrary OpenAI-compatible endpoints."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = require_non_empty(base_url, "base_url")
        self._api_key = require_non_empty(api_key, "api_key")

    async def generate_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        normalized_model = require_non_empty(model, "model")
        validate_prompt_inputs(system_prompt, user_prompt)
        payload = build_chat_payload(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=False,
            include_model=True,
        )
        url = _build_chat_completions_url(self._base_url)
        headers = _build_headers(self._api_key)
        logger.info("openai_compatible_chat_started model=%s url=%s", normalized_model, url)
        response = await self._post_json(url, headers, payload)
        if response.status_code != 200:
            raise RuntimeError(
                "OpenAI-compatible chat request failed: "
                f"{response.status_code} {response.text}"
            )
        content = extract_chat_content(
            response.json(),
            error_prefix="OpenAI-compatible chat response",
        )
        logger.info(
            "openai_compatible_chat_completed model=%s response_length=%s",
            normalized_model,
            len(content),
        )
        return content

    async def stream_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        normalized_model = require_non_empty(model, "model")
        validate_prompt_inputs(system_prompt, user_prompt)
        payload = build_chat_payload(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=True,
            include_model=True,
        )
        url = _build_chat_completions_url(self._base_url)
        headers = _build_headers(self._api_key)
        logger.info(
            "openai_compatible_chat_stream_started model=%s url=%s",
            normalized_model,
            url,
        )
        async for data_line in self._stream_post_data_lines(url, headers, payload):
            chunk = extract_stream_chunk(
                data_line,
                error_prefix="OpenAI-compatible chat stream event",
            )
            if chunk != "":
                yield chunk
        logger.info("openai_compatible_chat_stream_completed model=%s", normalized_model)

    async def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Any:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency for OpenAI-compatible chat provider: httpx"
            ) from exc

        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, headers=headers, json=payload)

    async def _stream_post_data_lines(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency for OpenAI-compatible chat provider: httpx"
            ) from exc

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode("utf-8")
                    raise RuntimeError(
                        "OpenAI-compatible chat stream request failed: "
                        f"{response.status_code} {error_body}"
                    )
                async for raw_line in response.aiter_lines():
                    if raw_line == "" or not raw_line.startswith("data: "):
                        continue
                    payload_line = raw_line[6:]
                    if payload_line == "[DONE]":
                        return
                    yield payload_line


def _build_chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
