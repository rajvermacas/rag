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


class FakeChatProviderRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id == "lab_vllm":
            return "openai_compatible"
        if backend_id == "azure_prod":
            return "azure_openai"
        raise ValueError("backend_id is not allowed")

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        self.calls.append((backend_id, model, system_prompt, user_prompt))
        if backend_id == "lab_vllm":
            yield "A1"
            yield "A2"
            return
        yield "B1"
        yield "B2"


class FakeChatProviderRouterWithSideError:
    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id == "lab_vllm":
            return "openai_compatible"
        if backend_id == "azure_prod":
            return "azure_openai"
        raise ValueError("backend_id is not allowed")

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        if backend_id == "lab_vllm":
            yield "A1"
            yield "A2"
            return
        yield "B1"
        raise RuntimeError("model-b stream failed")


class FakeChatProviderRouterNoEvidence:
    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id == "lab_vllm":
            return "openai_compatible"
        if backend_id == "azure_prod":
            return "azure_openai"
        raise ValueError("backend_id is not allowed")

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        if NO_DOCUMENT_EVIDENCE not in system_prompt:
            raise AssertionError("expected no-evidence guidance in system prompt")
        yield f"{backend_id}:{model}"


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
            model_a_backend_id="lab_vllm",
            model_a="model-a",
            model_b_backend_id="azure_prod",
            model_b="model-b",
        )
        return [event async for event in stream]

    return asyncio.run(collect())


def test_compare_stream_retrieves_once_and_forwards_backend_model() -> None:
    retrieval = FakeRetrievalService(_sample_chunks())
    chat_router = FakeChatProviderRouter()
    service = BattlegroundService(
        retrieval_service=retrieval,
        chat_client=chat_router,
        document_service=FakeDocumentService(),
    )

    events = _collect_events(service, "What is revenue?")

    assert retrieval.calls == 1
    assert retrieval.last_query is not None
    assert len(chat_router.calls) == 2
    assert chat_router.calls[0][0] == "lab_vllm"
    assert chat_router.calls[0][1] == "model-a"
    assert chat_router.calls[1][0] == "azure_prod"
    assert chat_router.calls[1][1] == "model-b"

    chunk_sides = {event.side for event in events if event.kind == "chunk"}
    done_sides = {event.side for event in events if event.kind == "done"}
    assert chunk_sides == {"A", "B"}
    assert done_sides == {"A", "B"}


def test_compare_stream_validation_errors_fail_fast() -> None:
    service = BattlegroundService(
        retrieval_service=FakeRetrievalService(_sample_chunks()),
        chat_client=FakeChatProviderRouter(),
        document_service=FakeDocumentService(),
    )

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
        asyncio.run(collect_invalid(" ", "lab_vllm", "model-a", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_a_backend_id must not be empty"):
        asyncio.run(collect_invalid("q", "  ", "model-a", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_a must not be empty"):
        asyncio.run(collect_invalid("q", "lab_vllm", "  ", "azure_prod", "model-b"))
    with pytest.raises(ValueError, match="model_b_backend_id must not be empty"):
        asyncio.run(collect_invalid("q", "lab_vllm", "model-a", " ", "model-b"))
    with pytest.raises(ValueError, match="model_b must not be empty"):
        asyncio.run(collect_invalid("q", "lab_vllm", "model-a", "azure_prod", " "))


def test_compare_stream_emits_side_specific_error_and_continues() -> None:
    service = BattlegroundService(
        retrieval_service=FakeRetrievalService(_sample_chunks()),
        chat_client=FakeChatProviderRouterWithSideError(),
        document_service=FakeDocumentService(),
    )

    events = _collect_events(service, "What is revenue?")

    error_events = [event for event in events if event.kind == "error"]
    done_events = [event for event in events if event.kind == "done"]
    assert len(error_events) == 1
    assert error_events[0].side == "B"
    assert error_events[0].error == "model-b stream failed"
    assert {event.side for event in done_events} == {"A"}


def test_compare_stream_handles_no_evidence_retrieval_like_chat() -> None:
    service = BattlegroundService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatProviderRouterNoEvidence(),
        document_service=FakeDocumentService(),
    )

    events = _collect_events(service, "What is revenue?")

    chunk_events = [event for event in events if event.kind == "chunk"]
    assert {event.side for event in chunk_events} == {"A", "B"}
