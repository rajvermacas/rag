import asyncio
from dataclasses import dataclass

from app.services.chat import ConversationTurn
from app.services.query_engine import QueryEngineService


@dataclass(frozen=True)
class FakeQueryResponse:
    response: str
    source_nodes: list[object]


class FakeQueryEngine:
    async def aquery(self, query: str) -> FakeQueryResponse:
        if "What is in my docs?" not in query:
            raise AssertionError("question was not included in query payload")
        return FakeQueryResponse(
            response="The docs mention revenue. [a.txt#0]",
            source_nodes=[object()],
        )


class FakeQueryEngineFactory:
    def __init__(self, stream_chunks: list[str] | None = None) -> None:
        self.last_llm: object | None = None
        self.stream_chunks = stream_chunks

    def build_query_engine(self, llm: object):
        self.last_llm = llm
        if self.stream_chunks is not None:
            return FakeStreamingQueryEngine(self.stream_chunks)
        return FakeQueryEngine()

    def build_streaming_query_engine(self, llm: object):
        self.last_llm = llm
        if self.stream_chunks is None:
            raise AssertionError("stream_chunks must be configured for streaming tests")
        return FakeStreamingQueryEngine(self.stream_chunks)


class FakeStreamingResponse:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def async_response_gen(self):
        for chunk in self._chunks:
            yield chunk


class FakeStreamingQueryEngine:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def aquery(self, query: str) -> FakeStreamingResponse:
        if query.strip() == "":
            raise AssertionError("query payload must not be empty")
        return FakeStreamingResponse(self._chunks)


class FakeLLMRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_llm(self, backend_id: str, model: str) -> object:
        self.calls.append((backend_id, model))
        return {"backend_id": backend_id, "model": model}


def build_query_engine_service() -> QueryEngineService:
    return QueryEngineService(
        llm_registry=FakeLLMRegistry(),
        engine_factory=FakeQueryEngineFactory(),
    )


def build_query_engine_service_with_stream(chunks: list[str]) -> QueryEngineService:
    return QueryEngineService(
        llm_registry=FakeLLMRegistry(),
        engine_factory=FakeQueryEngineFactory(stream_chunks=chunks),
    )


def test_answer_question_uses_backend_model_and_returns_grounded_flag() -> None:
    service = build_query_engine_service()

    result = asyncio.run(
        service.answer_question(
            question="What is in my docs?",
            history=[ConversationTurn(role="user", message="Earlier prompt")],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
    )

    assert isinstance(result.answer, str)
    assert isinstance(result.grounded, bool)


def test_stream_answer_question_yields_token_chunks_in_order() -> None:
    service = build_query_engine_service_with_stream(["Hello ", "world"])

    async def collect() -> list[str]:
        chunks = []
        async for chunk in service.stream_answer_question(
            question="hi",
            history=[],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert chunks == ["Hello ", "world"]
