"""Battleground service for fair side-by-side model comparison."""

import asyncio
from dataclasses import dataclass
import logging
from typing import AsyncIterator, Protocol

from app.services.chat import ConversationTurn, _validate_history


logger = logging.getLogger(__name__)
_COMPARE_STREAM_QUEUE_MAXSIZE = 64


class QueryService(Protocol):
    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Stream a response for a backend/model pair."""


@dataclass(frozen=True)
class CompareStreamEvent:
    side: str
    kind: str
    chunk: str | None
    error: str | None


class BattlegroundService:
    """Run side-by-side streaming comparisons through query-engine streaming."""

    def __init__(self, query_service: QueryService) -> None:
        self._query_service = query_service

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
        logger.info(
            "battleground_compare_started question_length=%s history_turns=%s "
            "model_a_backend_id=%s model_a=%s model_b_backend_id=%s model_b=%s",
            len(question),
            len(history),
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2],
            normalized_inputs[3],
        )
        queue: asyncio.Queue[CompareStreamEvent] = asyncio.Queue(
            maxsize=_COMPARE_STREAM_QUEUE_MAXSIZE
        )
        stream_a = self._query_service.stream_answer_question(
            question=question,
            history=history,
            backend_id=normalized_inputs[0],
            model=normalized_inputs[1],
        )
        stream_b = self._query_service.stream_answer_question(
            question=question,
            history=history,
            backend_id=normalized_inputs[2],
            model=normalized_inputs[3],
        )
        tasks = [
            asyncio.create_task(self._stream_side("A", stream_a, queue)),
            asyncio.create_task(self._stream_side("B", stream_b, queue)),
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

    async def _stream_side(
        self,
        side: str,
        stream: AsyncIterator[str],
        queue: asyncio.Queue[CompareStreamEvent],
    ) -> None:
        logger.info("battleground_side_stream_started side=%s", side)
        chunk_count = 0
        try:
            async for chunk in stream:
                if chunk == "":
                    continue
                chunk_count += 1
                await queue.put(CompareStreamEvent(side=side, kind="chunk", chunk=chunk, error=None))
            logger.info(
                "battleground_side_stream_completed side=%s chunk_count=%s",
                side,
                chunk_count,
            )
            await queue.put(CompareStreamEvent(side=side, kind="done", chunk=None, error=None))
        except BaseException as exc:
            if _is_current_task_cancellation(exc):
                logger.info("battleground_side_stream_cancelled side=%s", side)
                raise
            logger.exception(
                "battleground_side_stream_failed side=%s error=%s",
                side,
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
