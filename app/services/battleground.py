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


class RetrievalService(Protocol):
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        """Retrieve relevant chunks."""


class ChatClient(Protocol):
    async def stream_chat_response_with_model(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        """Stream a chat response from OpenRouter with a model override."""


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
        allowed_models: tuple[str, ...],
    ) -> None:
        _validate_allowed_models(allowed_models)
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client
        self._document_service = document_service
        self._allowed_models = allowed_models

    async def compare_stream(
        self,
        question: str,
        history: list[ConversationTurn],
        model_a: str,
        model_b: str,
    ) -> AsyncIterator[CompareStreamEvent]:
        _validate_compare_inputs(question, model_a, model_b, self._allowed_models)
        _validate_history(history)
        documents = self._document_service.list_documents()
        logger.info(
            "battleground_compare_started question_length=%s history_turns=%s model_a=%s "
            "model_b=%s available_documents=%s",
            len(question),
            len(history),
            model_a,
            model_b,
            len(documents),
        )
        retrieval_query = _build_retrieval_query(question, history)
        chunks = await self._retrieve_chunks_or_empty(retrieval_query)
        system_prompt = _build_system_prompt(len(chunks) > 0)
        user_prompt = _build_user_prompt(question, history, chunks, documents)
        queue: asyncio.Queue[CompareStreamEvent] = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                self._stream_model_side("A", model_a, system_prompt, user_prompt, queue)
            ),
            asyncio.create_task(
                self._stream_model_side("B", model_b, system_prompt, user_prompt, queue)
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
        model: str,
        system_prompt: str,
        user_prompt: str,
        queue: asyncio.Queue[CompareStreamEvent],
    ) -> None:
        logger.info("battleground_side_stream_started side=%s model=%s", side, model)
        chunk_count = 0
        try:
            async for chunk in self._chat_client.stream_chat_response_with_model(
                model,
                system_prompt,
                user_prompt,
            ):
                if chunk == "":
                    continue
                chunk_count += 1
                await queue.put(CompareStreamEvent(side=side, kind="chunk", chunk=chunk, error=None))
            logger.info(
                "battleground_side_stream_completed side=%s model=%s chunk_count=%s",
                side,
                model,
                chunk_count,
            )
            await queue.put(CompareStreamEvent(side=side, kind="done", chunk=None, error=None))
        except Exception as exc:
            logger.exception(
                "battleground_side_stream_failed side=%s model=%s error=%s",
                side,
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
    model_a: str,
    model_b: str,
    allowed_models: tuple[str, ...],
) -> None:
    if question.strip() == "":
        raise ValueError("question must not be empty")
    if model_a.strip() == "":
        raise ValueError("model_a must not be empty")
    if model_b.strip() == "":
        raise ValueError("model_b must not be empty")
    if model_a == model_b:
        raise ValueError("model_a and model_b must be different")
    if model_a not in allowed_models:
        raise ValueError("model_a is not allowed")
    if model_b not in allowed_models:
        raise ValueError("model_b is not allowed")


def _validate_allowed_models(allowed_models: tuple[str, ...]) -> None:
    if len(allowed_models) == 0:
        raise ValueError("allowed_models must not be empty")
    normalized = [model.strip() for model in allowed_models]
    if any(model == "" for model in normalized):
        raise ValueError("allowed_models contains empty model id")
    if len(set(normalized)) != len(normalized):
        raise ValueError("allowed_models contains duplicate model id")
