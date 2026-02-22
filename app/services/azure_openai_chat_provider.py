"""Provider client for Azure OpenAI chat completion APIs."""

import logging
from typing import Any, AsyncIterator, Mapping

from app.services.chat_provider_models import (
    build_chat_payload,
    extract_chat_content,
    extract_stream_chunk,
    require_non_empty,
    validate_prompt_inputs,
)


logger = logging.getLogger(__name__)


class AzureOpenAIChatProvider:
    """Chat provider for Azure OpenAI deployment-based endpoints."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployments: Mapping[str, str],
    ) -> None:
        self._endpoint = require_non_empty(endpoint, "endpoint")
        self._api_key = require_non_empty(api_key, "api_key")
        self._api_version = require_non_empty(api_version, "api_version")
        self._deployments = _validate_deployments(deployments)

    async def generate_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        normalized_model = require_non_empty(model, "model")
        validate_prompt_inputs(system_prompt, user_prompt)
        deployment = _resolve_deployment(normalized_model, self._deployments)
        payload = build_chat_payload(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=False,
            include_model=False,
        )
        url = _build_chat_completions_url(self._endpoint, deployment, self._api_version)
        headers = _build_headers(self._api_key)
        logger.info("azure_openai_chat_started model=%s deployment=%s", normalized_model, deployment)
        response = await self._post_json(url, headers, payload)
        if response.status_code != 200:
            raise RuntimeError(
                "Azure OpenAI chat request failed: "
                f"{response.status_code} {response.text}"
            )
        content = extract_chat_content(response.json(), error_prefix="Azure OpenAI chat response")
        logger.info(
            "azure_openai_chat_completed model=%s deployment=%s response_length=%s",
            normalized_model,
            deployment,
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
        deployment = _resolve_deployment(normalized_model, self._deployments)
        payload = build_chat_payload(
            model=normalized_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=True,
            include_model=False,
        )
        url = _build_chat_completions_url(self._endpoint, deployment, self._api_version)
        headers = _build_headers(self._api_key)
        logger.info(
            "azure_openai_chat_stream_started model=%s deployment=%s",
            normalized_model,
            deployment,
        )
        async for data_line in self._stream_post_data_lines(url, headers, payload):
            chunk = extract_stream_chunk(
                data_line,
                error_prefix="Azure OpenAI chat stream event",
            )
            if chunk != "":
                yield chunk
        logger.info(
            "azure_openai_chat_stream_completed model=%s deployment=%s",
            normalized_model,
            deployment,
        )

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
                "Missing dependency for Azure OpenAI chat provider: httpx"
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
                "Missing dependency for Azure OpenAI chat provider: httpx"
            ) from exc

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode("utf-8")
                    raise RuntimeError(
                        "Azure OpenAI chat stream request failed: "
                        f"{response.status_code} {error_body}"
                    )
                async for raw_line in response.aiter_lines():
                    if raw_line == "" or not raw_line.startswith("data: "):
                        continue
                    payload_line = raw_line[6:]
                    if payload_line == "[DONE]":
                        return
                    yield payload_line


def _validate_deployments(deployments: Mapping[str, str]) -> dict[str, str]:
    if len(deployments) == 0:
        raise ValueError("deployments must not be empty")
    validated_deployments: dict[str, str] = {}
    for model, deployment in deployments.items():
        normalized_model = require_non_empty(model, "model")
        normalized_deployment = require_non_empty(deployment, "deployment")
        validated_deployments[normalized_model] = normalized_deployment
    return validated_deployments


def _resolve_deployment(model: str, deployments: Mapping[str, str]) -> str:
    if model not in deployments:
        raise ValueError("model is not configured for azure deployment")
    return deployments[model]


def _build_chat_completions_url(endpoint: str, deployment: str, api_version: str) -> str:
    trimmed_endpoint = endpoint.rstrip("/")
    return (
        f"{trimmed_endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
