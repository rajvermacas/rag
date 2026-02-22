import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from app.services.chat import ChatResult, ChatService, ConversationTurn


@dataclass(frozen=True)
class FakeQueryResult:
    answer: str
    citations: list[dict[str, str | float | int | None]]
    grounded: bool
    retrieved_count: int


class FakeQueryService:
    def __init__(self) -> None:
        self.answer_calls: list[tuple[str, str, str]] = []
        self.stream_calls: list[tuple[str, str, str]] = []

    async def answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> FakeQueryResult:
        if len(history) == 0:
            raise AssertionError("history must be passed to query service")
        self.answer_calls.append((question, backend_id, model))
        return FakeQueryResult(
            answer="Revenue is 20.",
            citations=[],
            grounded=True,
            retrieved_count=1,
        )

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        if len(history) == 0:
            raise AssertionError("history must be passed to query stream service")
        self.stream_calls.append((question, backend_id, model))
        yield "Revenue "
        yield "is 20."


def test_chat_answer_delegates_to_query_service() -> None:
    query_service = FakeQueryService()
    service = ChatService(query_service=query_service)
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(
        service.answer_question(
            question="What is revenue?",
            history=history,
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
    )

    assert isinstance(result, ChatResult)
    assert result.answer == "Revenue is 20."
    assert result.grounded is True
    assert result.retrieved_count == 1
    assert query_service.answer_calls == [
        ("What is revenue?", "openrouter_lab", "openai/gpt-4o-mini")
    ]


def test_chat_stream_delegates_to_query_service() -> None:
    query_service = FakeQueryService()
    service = ChatService(query_service=query_service)
    history = [ConversationTurn(role="user", message="Earlier message")]

    async def collect_chunks() -> list[str]:
        stream = service.stream_answer_question(
            question="What is revenue?",
            history=history,
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect_chunks())

    assert chunks == ["Revenue ", "is 20."]
    assert query_service.stream_calls == [
        ("What is revenue?", "openrouter_lab", "openai/gpt-4o-mini")
    ]


def test_chat_rejects_empty_history_message() -> None:
    service = ChatService(query_service=FakeQueryService())

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=[ConversationTurn(role="assistant", message=" ")],
                backend_id="openrouter_lab",
                model="openai/gpt-4o-mini",
            )
        )
        raise AssertionError("expected ValueError for empty history message")
    except ValueError as exc:
        assert str(exc) == "history message must not be empty"


def test_chat_rejects_empty_backend_id() -> None:
    service = ChatService(query_service=FakeQueryService())

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=[ConversationTurn(role="user", message="Earlier")],
                backend_id="   ",
                model="openai/gpt-4o-mini",
            )
        )
        raise AssertionError("expected ValueError for empty backend_id")
    except ValueError as exc:
        assert str(exc) == "backend_id must not be empty"


def test_chat_rejects_empty_model_id() -> None:
    service = ChatService(query_service=FakeQueryService())

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=[ConversationTurn(role="user", message="Earlier")],
                backend_id="openrouter_lab",
                model="   ",
            )
        )
        raise AssertionError("expected ValueError for empty model")
    except ValueError as exc:
        assert str(exc) == "model must not be empty"
