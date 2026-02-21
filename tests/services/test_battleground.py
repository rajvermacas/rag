import asyncio

import pytest

from app.services.battleground import BattlegroundService
from app.services.chat import ConversationTurn, NO_DOCUMENT_EVIDENCE
from app.services.vector_store import IndexedChunk, IndexedDocument


class FakeRetrievalService:
    def __init__(self, chunks: list[IndexedChunk]) -> None:
        self.calls = 0
        self.last_query: str | None = None
        self._chunks = chunks

    async def retrieve(self, question: str) -> list[IndexedChunk]:
        self.calls += 1
        self.last_query = question
        return self._chunks


class FakeRetrievalNoEvidence:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, question: str) -> list[IndexedChunk]:
        self.calls += 1
        raise ValueError("no results passed relevance threshold")


class FakeDocumentService:
    def list_documents(self) -> list[IndexedDocument]:
        return [
            IndexedDocument(doc_id="doc-1", filename="a.txt", chunks_indexed=1),
            IndexedDocument(doc_id="doc-2", filename="b.pdf", chunks_indexed=2),
        ]


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def stream_chat_response_with_model(
        self, model: str, system_prompt: str, user_prompt: str
    ):
        self.calls.append((model, system_prompt, user_prompt))
        parts_by_model = {
            "model-a": ["A1", "A2"],
            "model-b": ["B1", "B2"],
        }
        for part in parts_by_model[model]:
            yield part


class FakeChatClientWithSideError:
    async def stream_chat_response_with_model(
        self, model: str, system_prompt: str, user_prompt: str
    ):
        if model == "model-a":
            yield "A1"
            yield "A2"
            return
        if model == "model-b":
            yield "B1"
            raise RuntimeError("model-b stream failed")


class FakeChatClientNoEvidence:
    async def stream_chat_response_with_model(
        self, model: str, system_prompt: str, user_prompt: str
    ):
        if NO_DOCUMENT_EVIDENCE not in system_prompt:
            raise AssertionError("expected no-evidence guidance in system prompt")
        if "Available uploaded documents:" not in user_prompt:
            raise AssertionError("expected uploaded documents section")
        if "Context:\n" not in user_prompt:
            raise AssertionError("expected context section")
        yield f"{model}-ok"


def _sample_chunks() -> list[IndexedChunk]:
    return [
        IndexedChunk(
            doc_id="doc-1",
            filename="a.txt",
            chunk_id="0",
            text="The answer is 20.",
            score=0.92,
            page=None,
        )
    ]


def _collect_events(service: BattlegroundService, question: str) -> list:
    async def collect() -> list:
        stream = service.compare_stream(
            question=question,
            history=[ConversationTurn(role="user", message="Earlier message")],
            model_a="model-a",
            model_b="model-b",
        )
        return [event async for event in stream]

    return asyncio.run(collect())


def test_compare_stream_retrieves_once_and_tags_sides() -> None:
    retrieval = FakeRetrievalService(_sample_chunks())
    chat_client = FakeChatClient()
    service = BattlegroundService(
        retrieval_service=retrieval,
        chat_client=chat_client,
        document_service=FakeDocumentService(),
        allowed_models=("model-a", "model-b"),
    )

    events = _collect_events(service, "What is revenue?")

    assert retrieval.calls == 1
    assert retrieval.last_query is not None
    assert "Conversation history:" in retrieval.last_query
    assert "Current question:" in retrieval.last_query
    assert len(chat_client.calls) == 2
    assert chat_client.calls[0][1] == chat_client.calls[1][1]
    assert chat_client.calls[0][2] == chat_client.calls[1][2]

    chunk_sides = {event.side for event in events if event.kind == "chunk"}
    done_sides = {event.side for event in events if event.kind == "done"}
    assert chunk_sides == {"A", "B"}
    assert done_sides == {"A", "B"}


def test_compare_stream_validation_errors_fail_fast() -> None:
    retrieval = FakeRetrievalService(_sample_chunks())
    service = BattlegroundService(
        retrieval_service=retrieval,
        chat_client=FakeChatClient(),
        document_service=FakeDocumentService(),
        allowed_models=("model-a", "model-b"),
    )

    async def collect_invalid(question: str, model_a: str, model_b: str) -> list:
        stream = service.compare_stream(
            question=question,
            history=[],
            model_a=model_a,
            model_b=model_b,
        )
        return [event async for event in stream]

    with pytest.raises(ValueError, match="question must not be empty"):
        asyncio.run(collect_invalid(" ", "model-a", "model-b"))
    with pytest.raises(ValueError, match="model_a must not be empty"):
        asyncio.run(collect_invalid("q", "  ", "model-b"))
    with pytest.raises(ValueError, match="model_b must not be empty"):
        asyncio.run(collect_invalid("q", "model-a", " "))
    with pytest.raises(ValueError, match="model_a and model_b must be different"):
        asyncio.run(collect_invalid("q", "model-a", "model-a"))
    with pytest.raises(ValueError, match="model_a is not allowed"):
        asyncio.run(collect_invalid("q", "bad-model", "model-b"))
    with pytest.raises(ValueError, match="model_b is not allowed"):
        asyncio.run(collect_invalid("q", "model-a", "bad-model"))
    assert retrieval.calls == 0


def test_compare_stream_emits_side_specific_error_and_continues() -> None:
    service = BattlegroundService(
        retrieval_service=FakeRetrievalService(_sample_chunks()),
        chat_client=FakeChatClientWithSideError(),
        document_service=FakeDocumentService(),
        allowed_models=("model-a", "model-b"),
    )

    events = _collect_events(service, "What is revenue?")

    error_events = [event for event in events if event.kind == "error"]
    done_events = [event for event in events if event.kind == "done"]
    assert len(error_events) == 1
    assert error_events[0].side == "B"
    assert error_events[0].error == "model-b stream failed"
    assert {event.side for event in done_events} == {"A"}
    assert any(event.kind == "chunk" and event.side == "A" for event in events)


def test_compare_stream_handles_no_evidence_retrieval_like_chat() -> None:
    service = BattlegroundService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
        document_service=FakeDocumentService(),
        allowed_models=("model-a", "model-b"),
    )

    events = _collect_events(service, "What is revenue?")

    chunk_events = [event for event in events if event.kind == "chunk"]
    assert {event.side for event in chunk_events} == {"A", "B"}
