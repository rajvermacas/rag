import asyncio
from dataclasses import dataclass

from app.services.chat import ConversationTurn
from app.services.query_engine import EMPTY_STREAM_RESPONSE_TEXT, QueryEngineService


@dataclass(frozen=True)
class FakeQueryResponse:
    response: str
    source_nodes: list[object]


@dataclass(frozen=True)
class FakeCompletion:
    text: str


class FakeQueryEngine:
    async def aquery(self, query: str) -> FakeQueryResponse:
        if "empty sentinel" in query:
            return FakeQueryResponse(
                response="Empty Response",
                source_nodes=[],
            )
        if "citation only" in query:
            return FakeQueryResponse(
                response="[a.txt#0]",
                source_nodes=[object()],
            )
        if "What is in my docs?" not in query:
            raise AssertionError("question was not included in query payload")
        return FakeQueryResponse(
            response="The docs mention revenue. [a.txt#0]",
            source_nodes=[object()],
        )


class FakeQueryEngineFactory:
    def __init__(
        self,
        stream_chunks: list[str] | None = None,
        has_documents: bool = True,
    ) -> None:
        self.last_llm: object | None = None
        self.stream_chunks = stream_chunks
        self.has_documents = has_documents
        self.query_engine_build_count = 0
        self.streaming_query_engine_build_count = 0

    def build_query_engine(self, llm: object):
        self.last_llm = llm
        self.query_engine_build_count += 1
        if self.stream_chunks is not None:
            return FakeStreamingQueryEngine(self.stream_chunks)
        return FakeQueryEngine()

    def build_streaming_query_engine(self, llm: object):
        self.last_llm = llm
        self.streaming_query_engine_build_count += 1
        if self.stream_chunks is None:
            raise AssertionError("stream_chunks must be configured for streaming tests")
        return FakeStreamingQueryEngine(self.stream_chunks)

    def has_indexed_documents(self) -> bool:
        return self.has_documents


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
    def __init__(self, general_chat_answer: str = "Hello from general knowledge.") -> None:
        self.calls: list[tuple[str, str]] = []
        self.general_chat_answer = general_chat_answer

    def get_llm(self, backend_id: str, model: str) -> object:
        self.calls.append((backend_id, model))
        return FakeLLM(self.general_chat_answer)


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def acomplete(self, prompt: str) -> FakeCompletion:
        if prompt.strip() == "":
            raise AssertionError("general chat prompt must not be empty")
        return FakeCompletion(text=self._answer)


def build_query_engine_service(
    has_documents: bool = True,
    general_chat_answer: str = "Hello from general knowledge.",
) -> QueryEngineService:
    return QueryEngineService(
        llm_registry=FakeLLMRegistry(general_chat_answer=general_chat_answer),
        engine_factory=FakeQueryEngineFactory(has_documents=has_documents),
    )


def build_query_engine_service_with_stream(
    chunks: list[str],
    has_documents: bool = True,
    general_chat_answer: str = "Hello from general knowledge.",
) -> QueryEngineService:
    return QueryEngineService(
        llm_registry=FakeLLMRegistry(general_chat_answer=general_chat_answer),
        engine_factory=FakeQueryEngineFactory(
            stream_chunks=chunks,
            has_documents=has_documents,
        ),
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


def test_answer_question_replaces_citation_only_text_with_fallback() -> None:
    service = build_query_engine_service()

    result = asyncio.run(
        service.answer_question(
            question="citation only",
            history=[ConversationTurn(role="user", message="Earlier prompt")],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.answer == EMPTY_STREAM_RESPONSE_TEXT


def test_answer_question_replaces_empty_response_sentinel_with_fallback() -> None:
    service = build_query_engine_service()

    result = asyncio.run(
        service.answer_question(
            question="empty sentinel",
            history=[ConversationTurn(role="user", message="Earlier prompt")],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.answer == EMPTY_STREAM_RESPONSE_TEXT


def test_answer_question_uses_general_chat_when_index_has_no_documents() -> None:
    service = build_query_engine_service(
        has_documents=False,
        general_chat_answer="Hi there! How can I help?",
    )

    result = asyncio.run(
        service.answer_question(
            question="hi",
            history=[],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.answer == "Hi there! How can I help?"
    assert result.grounded is False
    assert result.retrieved_count == 0


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


def test_stream_answer_question_preserves_newline_chunks_for_markdown_lists() -> None:
    service = build_query_engine_service_with_stream(
        [
            "The policy is called HDFC Life Click2Achieve.",
            "\n",
            "- **Policy term:** 30 years",
            "\n",
            "- **Premium-paying term:** 12 years",
        ]
    )

    async def collect() -> list[str]:
        chunks = []
        async for chunk in service.stream_answer_question(
            question="Summarize this policy",
            history=[],
            backend_id="openrouter_lab",
            model="openai/gpt-4o-mini",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert chunks == [
        "The policy is called HDFC Life Click2Achieve.",
        "\n",
        "- **Policy term:** 30 years",
        "\n",
        "- **Premium-paying term:** 12 years",
    ]


def test_stream_answer_question_replaces_empty_stream_with_message() -> None:
    service = build_query_engine_service_with_stream(["", "   "])

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

    assert chunks == [EMPTY_STREAM_RESPONSE_TEXT]


def test_stream_answer_question_appends_fallback_for_citation_only_stream() -> None:
    service = build_query_engine_service_with_stream(["[a.txt#0]"])

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

    assert chunks == [EMPTY_STREAM_RESPONSE_TEXT]


def test_stream_answer_question_replaces_empty_response_sentinel() -> None:
    service = build_query_engine_service_with_stream(["Empty Response"])

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

    assert chunks == [EMPTY_STREAM_RESPONSE_TEXT]


def test_stream_answer_question_uses_general_chat_when_index_has_no_documents() -> None:
    service = build_query_engine_service_with_stream(
        ["ignored chunk"],
        has_documents=False,
        general_chat_answer="Hi there! How can I help?",
    )

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

    assert chunks == ["Hi there! How can I help?"]


def test_warm_up_caches_llm_and_query_engines_per_backend_model() -> None:
    llm_registry = FakeLLMRegistry()
    engine_factory = FakeQueryEngineFactory(stream_chunks=["warmup"])
    service = QueryEngineService(
        llm_registry=llm_registry,
        engine_factory=engine_factory,
    )

    service.warm_up(
        (
            ("openrouter_lab", "openai/gpt-4o-mini"),
            ("openrouter_lab", "openai/gpt-4o-mini"),
        )
    )

    assert llm_registry.calls == [("openrouter_lab", "openai/gpt-4o-mini")]
    assert engine_factory.query_engine_build_count == 1
    assert engine_factory.streaming_query_engine_build_count == 1
