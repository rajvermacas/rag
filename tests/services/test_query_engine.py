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
    def __init__(self) -> None:
        self.last_llm: object | None = None

    def build_query_engine(self, llm: object):
        self.last_llm = llm
        return FakeQueryEngine()


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
