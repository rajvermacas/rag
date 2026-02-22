import asyncio

import pytest

from app.services.battleground import BattlegroundService
from app.services.chat import ConversationTurn


class FakeQueryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ):
        if question.strip() == "":
            raise AssertionError("question must not be empty")
        if len(history) == 0:
            raise AssertionError("history must be forwarded")
        self.calls.append((backend_id, model))
        if backend_id == "openrouter_lab":
            yield "A1"
            yield "A2"
            return
        yield "B1"
        yield "B2"


class FakeQueryServiceWithSideError:
    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ):
        if backend_id == "openrouter_lab":
            yield "A1"
            return
        raise RuntimeError("model-b stream failed")


def _collect_events(service: BattlegroundService, question: str) -> list:
    async def collect() -> list:
        stream = service.compare_stream(
            question=question,
            history=[ConversationTurn(role="user", message="Earlier message")],
            model_a_backend_id="openrouter_lab",
            model_a="openai/gpt-4o-mini",
            model_b_backend_id="azure_prod",
            model_b="gpt-4o-mini",
        )
        return [event async for event in stream]

    return asyncio.run(collect())


def test_compare_stream_emits_left_and_right_chunks() -> None:
    query_service = FakeQueryService()
    service = BattlegroundService(query_service=query_service)

    events = _collect_events(service, "question")

    chunk_sides = {event.side for event in events if event.kind == "chunk"}
    assert chunk_sides == {"A", "B"}
    assert query_service.calls == [
        ("openrouter_lab", "openai/gpt-4o-mini"),
        ("azure_prod", "gpt-4o-mini"),
    ]


def test_compare_stream_validation_errors_fail_fast() -> None:
    service = BattlegroundService(query_service=FakeQueryService())

    async def collect_invalid(
        question: str,
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ) -> list:
        stream = service.compare_stream(
            question=question,
            history=[],
            model_a_backend_id=model_a_backend_id,
            model_a=model_a,
            model_b_backend_id=model_b_backend_id,
            model_b=model_b,
        )
        return [event async for event in stream]

    with pytest.raises(ValueError, match="question must not be empty"):
        asyncio.run(collect_invalid(" ", "openrouter_lab", "model-a", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_a_backend_id must not be empty"):
        asyncio.run(collect_invalid("q", "  ", "model-a", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_a must not be empty"):
        asyncio.run(collect_invalid("q", "openrouter_lab", "  ", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_b_backend_id must not be empty"):
        asyncio.run(collect_invalid("q", "openrouter_lab", "model-a", " ", "model-b"))
    with pytest.raises(ValueError, match="model_b must not be empty"):
        asyncio.run(collect_invalid("q", "openrouter_lab", "model-a", "azure_prod", " "))


def test_compare_stream_emits_side_specific_error_and_continues() -> None:
    service = BattlegroundService(query_service=FakeQueryServiceWithSideError())

    events = _collect_events(service, "What is revenue?")

    error_events = [event for event in events if event.kind == "error"]
    done_events = [event for event in events if event.kind == "done"]
    assert len(error_events) == 1
    assert error_events[0].side == "B"
    assert error_events[0].error == "model-b stream failed"
    assert {event.side for event in done_events} == {"A"}
