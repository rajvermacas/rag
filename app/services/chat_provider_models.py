"""Shared provider models and payload helpers for chat providers."""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


def require_non_empty(value: str, field_name: str) -> str:
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized_value


def validate_prompt_inputs(system_prompt: str, user_prompt: str) -> None:
    require_non_empty(system_prompt, "system_prompt")
    require_non_empty(user_prompt, "user_prompt")


def build_chat_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    stream: bool,
    include_model: bool,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload: dict[str, Any] = {"stream": stream, "messages": messages}
    if include_model:
        payload["model"] = model
    return payload


def extract_chat_content(payload: dict[str, Any], error_prefix: str) -> str:
    if "choices" not in payload:
        raise ValueError(f"{error_prefix} missing choices")
    choices = payload["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(f"{error_prefix} missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict) or "message" not in first_choice:
        raise ValueError(f"{error_prefix} missing message")
    message = first_choice["message"]
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError(f"{error_prefix} missing message content")
    content = message["content"]
    if not isinstance(content, str):
        raise ValueError(f"{error_prefix} content must be a string")
    return content


def extract_stream_chunk(data_line: str, error_prefix: str) -> str:
    payload = json.loads(data_line)
    if "choices" not in payload:
        raise ValueError(f"{error_prefix} missing choices")
    choices = payload["choices"]
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError(f"{error_prefix} missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(f"{error_prefix} choice must be an object")
    if "delta" not in first_choice:
        return ""
    delta = first_choice["delta"]
    if not isinstance(delta, dict):
        raise ValueError(f"{error_prefix} delta must be an object")
    if "content" not in delta:
        return ""
    content = delta["content"]
    if not isinstance(content, str):
        raise ValueError(f"{error_prefix} content must be a string")
    return content


@dataclass(frozen=True)
class ChatModelOption:
    backend_id: str
    provider: str
    model: str
    label: str


class BackendChatProvider(Protocol):
    async def generate_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a non-streaming chat response for one model."""

    async def stream_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response for one model."""
