"""Chat service facade over query-engine orchestration."""

from dataclasses import dataclass
import logging
from typing import Any, AsyncIterator, Protocol

from app.services.vector_store import IndexedChunk, IndexedDocument


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
        return self._query_service.stream_answer_question(
            question=normalized_question,
            history=history,
            backend_id=normalized_backend_id,
            model=normalized_model,
        )


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


def _build_retrieval_query(question: str, history: list[ConversationTurn]) -> str:
    if len(history) == 0:
        return question
    history_lines = [f"{turn.role}: {turn.message}" for turn in history[-6:]]
    return f"Conversation history:\n{'\n'.join(history_lines)}\n\nCurrent question:\n{question}"


def _build_system_prompt(has_document_evidence: bool) -> str:
    if has_document_evidence:
        evidence_instruction = (
            "Use document evidence when available and do not include inline citations."
        )
    else:
        evidence_instruction = (
            f"If no evidence is available, state exactly: '{NO_DOCUMENT_EVIDENCE}'."
        )
    return (
        "You are a conversational retrieval assistant. Keep responses concise. "
        f"{evidence_instruction}"
    )


def _build_user_prompt(
    question: str,
    history: list[ConversationTurn],
    chunks: list[IndexedChunk],
    documents: list[IndexedDocument],
) -> str:
    history_text = _format_history(history)
    context_text = _format_context(chunks)
    documents_text = _format_documents(documents)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Available uploaded documents:\n{documents_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}"
    )


def _format_history(history: list[ConversationTurn]) -> str:
    if len(history) == 0:
        return "[none]"
    return "\n".join([f"{turn.role}: {turn.message}" for turn in history])


def _format_context(chunks: list[IndexedChunk]) -> str:
    if len(chunks) == 0:
        return "[none]"
    return "\n\n".join(
        [
            (
                f"[doc_id={chunk.doc_id} filename={chunk.filename} "
                f"chunk_id={chunk.chunk_id} score={chunk.score:.4f} page={chunk.page}]\n"
                f"{chunk.text}"
            )
            for chunk in chunks
        ]
    )


def _format_documents(documents: list[IndexedDocument]) -> str:
    if len(documents) == 0:
        return "[none]"
    return "\n".join(
        [
            f"{index + 1}. {document.filename} (doc_id={document.doc_id}, "
            f"chunks={document.chunks_indexed})"
            for index, document in enumerate(documents)
        ]
    )
