"""Chat service facade over query-engine orchestration."""

from dataclasses import dataclass
import logging
from typing import Any, AsyncIterator, Protocol

logger = logging.getLogger(__name__)
NO_DOCUMENT_EVIDENCE = "No relevant evidence found in uploaded documents."


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: list[dict[str, str | float | int | None]]
    grounded: bool
    retrieved_count: int


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    message: str


class QueryService(Protocol):
    async def answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> Any:
        """Execute non-streaming question answering."""

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Execute streaming question answering."""


class ChatService:
    """Validates chat inputs and delegates to query-engine services."""

    def __init__(self, query_service: QueryService) -> None:
        self._query_service = query_service

    async def answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> ChatResult:
        normalized_question = _require_non_empty(question, "question")
        normalized_backend_id = _require_non_empty(backend_id, "backend_id")
        normalized_model = _require_non_empty(model, "model")
        _validate_history(history)
        result = await self._query_service.answer_question(
            question=normalized_question,
            history=history,
            backend_id=normalized_backend_id,
            model=normalized_model,
        )
        logger.info(
            "chat_service_answer_completed backend_id=%s model=%s",
            normalized_backend_id,
            normalized_model,
        )
        return _to_chat_result(result)

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        normalized_question = _require_non_empty(question, "question")
        normalized_backend_id = _require_non_empty(backend_id, "backend_id")
        normalized_model = _require_non_empty(model, "model")
        _validate_history(history)
        logger.info(
            "chat_service_stream_started backend_id=%s model=%s",
            normalized_backend_id,
            normalized_model,
        )
        stream = self._query_service.stream_answer_question(
            question=normalized_question,
            history=history,
            backend_id=normalized_backend_id,
            model=normalized_model,
        )
        if not hasattr(stream, "__aiter__"):
            raise ValueError("query_service stream_answer_question must return an async iterator")
        async for chunk in stream:
            yield chunk


def _to_chat_result(result: Any) -> ChatResult:
    required_fields = ("answer", "citations", "grounded", "retrieved_count")
    missing_fields = [field for field in required_fields if not hasattr(result, field)]
    if len(missing_fields) > 0:
        raise ValueError(
            "query_service answer result is missing required fields: "
            f"{', '.join(missing_fields)}"
        )
    answer = getattr(result, "answer")
    citations = getattr(result, "citations")
    grounded = getattr(result, "grounded")
    retrieved_count = getattr(result, "retrieved_count")
    if not isinstance(answer, str):
        raise ValueError("query_service answer must be a string")
    if not isinstance(citations, list):
        raise ValueError("query_service citations must be a list")
    if not isinstance(grounded, bool):
        raise ValueError("query_service grounded must be a boolean")
    if not isinstance(retrieved_count, int):
        raise ValueError("query_service retrieved_count must be an integer")
    return ChatResult(
        answer=answer,
        citations=citations,
        grounded=grounded,
        retrieved_count=retrieved_count,
    )


def _validate_history(history: list[ConversationTurn]) -> None:
    for turn in history:
        if turn.role not in {"user", "assistant"}:
            raise ValueError("history role must be either 'user' or 'assistant'")
        _require_non_empty(turn.message, "history message")


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized
