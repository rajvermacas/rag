"""Battleground service for fair side-by-side model comparison."""

import asyncio
from dataclasses import dataclass
import logging
from typing import AsyncIterator, Protocol

from app.services.chat import (
    ConversationTurn,
    _build_retrieval_query,
    _build_system_prompt,
    _build_user_prompt,
    _validate_history,
)
from app.services.vector_store import IndexedChunk, IndexedDocument


logger = logging.getLogger(__name__)
_COMPARE_STREAM_QUEUE_MAXSIZE = 64


class RetrievalService(Protocol):
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        """Retrieve relevant chunks."""


class ChatClient(Protocol):
    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream a chat response with explicit backend and model selection."""

    def get_provider_for_backend(self, backend_id: str) -> str:
        """Resolve provider id for a backend id."""


class DocumentService(Protocol):
    def list_documents(self) -> list[IndexedDocument]:
        """Return all indexed documents."""


@dataclass(frozen=True)
class CompareStreamEvent:
    side: str
    kind: str
    chunk: str | None
    error: str | None


class BattlegroundService:
    """Run a fair compare stream for two models using shared retrieval and prompts."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        chat_client: ChatClient,
        document_service: DocumentService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client
        self._document_service = document_service

    async def compare_stream(
        self,
        question: str,
        history: list[ConversationTurn],
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ) -> AsyncIterator[CompareStreamEvent]:
        normalized_inputs = _validate_compare_inputs(
            question=question,
            model_a_backend_id=model_a_backend_id,
            model_a=model_a,
            model_b_backend_id=model_b_backend_id,
            model_b=model_b,
        )
        _validate_history(history)
        provider_a = self._chat_client.get_provider_for_backend(normalized_inputs[0])
        provider_b = self._chat_client.get_provider_for_backend(normalized_inputs[2])
        documents = self._document_service.list_documents()
        logger.info(
            "battleground_compare_started question_length=%s history_turns=%s "
            "model_a_backend_id=%s model_a_provider=%s model_a=%s "
            "model_b_backend_id=%s model_b_provider=%s model_b=%s "
            "available_documents=%s",
            len(question),
            len(history),
            normalized_inputs[0],
            provider_a,
            normalized_inputs[1],
            normalized_inputs[2],
            provider_b,
            normalized_inputs[3],
            len(documents),
        )
        retrieval_query = _build_retrieval_query(question, history)
        chunks = await self._retrieve_chunks_or_empty(retrieval_query)
        system_prompt = _build_system_prompt(len(chunks) > 0)
        user_prompt = _build_user_prompt(question, history, chunks, documents)
        queue: asyncio.Queue[CompareStreamEvent] = asyncio.Queue(
            maxsize=_COMPARE_STREAM_QUEUE_MAXSIZE
        )
        tasks = [
            asyncio.create_task(
                self._stream_model_side(
                    side="A",
                    backend_id=normalized_inputs[0],
                    provider=provider_a,
                    model=normalized_inputs[1],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    queue=queue,
                )
            ),
            asyncio.create_task(
                self._stream_model_side(
                    side="B",
                    backend_id=normalized_inputs[2],
                    provider=provider_b,
                    model=normalized_inputs[3],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    queue=queue,
                )
            ),
        ]
        terminal_events = 0
        try:
            while terminal_events < 2:
                event = await queue.get()
                if event.kind in {"done", "error"}:
                    terminal_events += 1
                yield event
        finally:
            await _cancel_pending_tasks(tasks)

    async def _retrieve_chunks_or_empty(self, question: str) -> list[IndexedChunk]:
        try:
            chunks = await self._retrieval_service.retrieve(question)
            logger.info("battleground_compare_retrieval_completed chunk_count=%s", len(chunks))
            return chunks
        except ValueError as exc:
            if str(exc) not in {
                "retrieval returned no results",
                "no results passed relevance threshold",
            }:
                raise
            logger.info("battleground_compare_no_evidence reason=%s", str(exc))
            return []

    async def _stream_model_side(
        self,
        side: str,
        backend_id: str,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        queue: asyncio.Queue[CompareStreamEvent],
    ) -> None:
        logger.info(
            "battleground_side_stream_started side=%s backend_id=%s provider=%s model=%s",
            side,
            backend_id,
            provider,
            model,
        )
        chunk_count = 0
        try:
            async for chunk in self._chat_client.stream_chat_response_with_backend(
                backend_id,
                model,
                system_prompt,
                user_prompt,
            ):
                if chunk == "":
                    continue
                chunk_count += 1
                await queue.put(CompareStreamEvent(side=side, kind="chunk", chunk=chunk, error=None))
            logger.info(
                "battleground_side_stream_completed side=%s backend_id=%s provider=%s "
                "model=%s chunk_count=%s",
                side,
                backend_id,
                provider,
                model,
                chunk_count,
            )
            await queue.put(CompareStreamEvent(side=side, kind="done", chunk=None, error=None))
        except BaseException as exc:
            if _is_current_task_cancellation(exc):
                logger.info(
                    "battleground_side_stream_cancelled side=%s backend_id=%s provider=%s "
                    "model=%s",
                    side,
                    backend_id,
                    provider,
                    model,
                )
                raise
            logger.exception(
                "battleground_side_stream_failed side=%s backend_id=%s provider=%s "
                "model=%s error=%s",
                side,
                backend_id,
                provider,
                model,
                str(exc),
            )
            await queue.put(
                CompareStreamEvent(side=side, kind="error", chunk=None, error=str(exc))
            )


async def _cancel_pending_tasks(tasks: list[asyncio.Task[None]]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if len(pending) == 0:
        return
    await asyncio.gather(*pending, return_exceptions=True)


def _validate_compare_inputs(
    question: str,
    model_a_backend_id: str,
    model_a: str,
    model_b_backend_id: str,
    model_b: str,
) -> tuple[str, str, str, str]:
    normalized_model_a_backend_id = _require_non_empty_field(
        model_a_backend_id,
        "model_a_backend_id",
    )
    normalized_model_a = _require_non_empty_field(model_a, "model_a")
    normalized_model_b_backend_id = _require_non_empty_field(
        model_b_backend_id,
        "model_b_backend_id",
    )
    normalized_model_b = _require_non_empty_field(model_b, "model_b")
    if question.strip() == "":
        raise ValueError("question must not be empty")
    if (
        normalized_model_a_backend_id == normalized_model_b_backend_id
        and normalized_model_a == normalized_model_b
    ):
        raise ValueError("model_a and model_b must be different")
    return (
        normalized_model_a_backend_id,
        normalized_model_a,
        normalized_model_b_backend_id,
        normalized_model_b,
    )


def _require_non_empty_field(value: str, field_name: str) -> str:
    normalized_value = value.strip()
    if normalized_value == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized_value


def _is_current_task_cancellation(exc: BaseException) -> bool:
    if not isinstance(exc, asyncio.CancelledError):
        return False
    task = asyncio.current_task()
    if task is None:
        return False
    return task.cancelling() > 0
