"""LlamaIndex-backed query orchestration for non-streaming chat."""

from dataclasses import dataclass
import logging
import re
from typing import Any, AsyncIterator, Protocol


logger = logging.getLogger(__name__)
INLINE_CITATION_PATTERN = re.compile(
    r"\[[^\]\n]*(?:#chunk_id\s*=\s*\d+|#\d+)[^\]\n]*\]",
    re.IGNORECASE,
)
EMPTY_STREAM_RESPONSE_TEXT = (
    "I could not generate a response from the current context. "
    "Please rephrase your question."
)
EMPTY_RESPONSE_SENTINELS = frozenset({"empty response", "empty reponse"})
BackendModelKey = tuple[str, str]


class LLMRegistry(Protocol):
    def get_llm(self, backend_id: str, model: str) -> Any:
        """Resolve validated backend/model to a configured LLM."""


class AsyncQueryEngine(Protocol):
    async def aquery(self, query: str) -> Any:
        """Execute asynchronous query."""


class QueryEngineFactory(Protocol):
    def build_query_engine(self, llm: Any) -> AsyncQueryEngine:
        """Build a query engine for the provided LLM."""

    def build_streaming_query_engine(self, llm: Any) -> AsyncQueryEngine:
        """Build a streaming-enabled query engine for the provided LLM."""


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[dict[str, str | float | int | None]]
    grounded: bool
    retrieved_count: int


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    message: str


class QueryEngineService:
    """Validate query inputs and execute LlamaIndex non-streaming queries."""

    def __init__(
        self,
        llm_registry: LLMRegistry,
        engine_factory: QueryEngineFactory,
    ) -> None:
        if llm_registry is None:
            raise ValueError("llm_registry must not be None")
        if engine_factory is None:
            raise ValueError("engine_factory must not be None")
        self._llm_registry = llm_registry
        self._engine_factory = engine_factory
        self._llm_cache: dict[BackendModelKey, Any] = {}
        self._query_engine_cache: dict[BackendModelKey, AsyncQueryEngine] = {}
        self._streaming_query_engine_cache: dict[BackendModelKey, AsyncQueryEngine] = {}

    def warm_up(self, backend_models: tuple[BackendModelKey, ...]) -> None:
        if len(backend_models) == 0:
            raise ValueError("backend_models must not be empty")
        logger.info("query_engine_warmup_started backend_model_count=%s", len(backend_models))
        for backend_id, model in backend_models:
            self._warm_up_backend_model(backend_id, model)
        logger.info("query_engine_warmup_completed backend_model_count=%s", len(backend_models))

    async def answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> QueryResult:
        normalized_question = _require_non_empty(question, "question")
        normalized_backend_id = _require_non_empty(backend_id, "backend_id")
        normalized_model = _require_non_empty(model, "model")
        _validate_history(history)
        engine = self._get_or_build_query_engine(normalized_backend_id, normalized_model)
        query_payload = _build_query(normalized_question, history)
        response = await engine.aquery(query_payload)
        result = _to_query_result(response)
        logger.info(
            "query_engine_answer_completed backend_id=%s model=%s grounded=%s "
            "retrieved_count=%s",
            normalized_backend_id,
            normalized_model,
            result.grounded,
            result.retrieved_count,
        )
        return result

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        normalized_question = _require_non_empty(question, "question")
        normalized_backend_id = _require_non_empty(backend_id, "backend_id")
        normalized_model = _require_non_empty(model, "model")
        _validate_history(history)
        engine = self._get_or_build_streaming_query_engine(
            normalized_backend_id,
            normalized_model,
        )
        query_payload = _build_query(normalized_question, history)
        response = await engine.aquery(query_payload)
        logger.info(
            "query_engine_stream_started backend_id=%s model=%s",
            normalized_backend_id,
            normalized_model,
        )
        emitted_chunk_count = 0
        emitted_visible_chunk_count = 0
        async for chunk in _iter_response_chunks(response):
            normalized_chunk = chunk.strip()
            if normalized_chunk == "":
                continue
            emitted_chunk_count += 1
            visible_chunk = _remove_inline_citations(chunk).strip()
            if _is_effectively_empty_response_text(visible_chunk):
                continue
            emitted_visible_chunk_count += 1
            yield chunk
        if emitted_visible_chunk_count == 0:
            logger.warning(
                "query_engine_stream_empty_visible_response backend_id=%s model=%s",
                normalized_backend_id,
                normalized_model,
            )
            yield EMPTY_STREAM_RESPONSE_TEXT
        logger.info(
            "query_engine_stream_completed backend_id=%s model=%s chunk_count=%s "
            "visible_chunk_count=%s",
            normalized_backend_id,
            normalized_model,
            emitted_chunk_count,
            emitted_visible_chunk_count,
        )

    def _warm_up_backend_model(self, backend_id: str, model: str) -> None:
        normalized_backend_id = _require_non_empty(backend_id, "backend_id")
        normalized_model = _require_non_empty(model, "model")
        self._get_or_build_query_engine(normalized_backend_id, normalized_model)
        self._get_or_build_streaming_query_engine(normalized_backend_id, normalized_model)
        logger.info(
            "query_engine_warmup_backend_model_completed backend_id=%s model=%s",
            normalized_backend_id,
            normalized_model,
        )

    def _get_or_build_query_engine(self, backend_id: str, model: str) -> AsyncQueryEngine:
        key = _build_backend_model_key(backend_id, model)
        if key in self._query_engine_cache:
            return self._query_engine_cache[key]
        llm = self._get_or_build_llm(backend_id, model)
        query_engine = self._engine_factory.build_query_engine(llm)
        self._query_engine_cache[key] = query_engine
        logger.info("query_engine_cached backend_id=%s model=%s", backend_id, model)
        return query_engine

    def _get_or_build_streaming_query_engine(
        self,
        backend_id: str,
        model: str,
    ) -> AsyncQueryEngine:
        key = _build_backend_model_key(backend_id, model)
        if key in self._streaming_query_engine_cache:
            return self._streaming_query_engine_cache[key]
        llm = self._get_or_build_llm(backend_id, model)
        streaming_query_engine = self._engine_factory.build_streaming_query_engine(llm)
        self._streaming_query_engine_cache[key] = streaming_query_engine
        logger.info("query_engine_stream_cached backend_id=%s model=%s", backend_id, model)
        return streaming_query_engine

    def _get_or_build_llm(self, backend_id: str, model: str) -> Any:
        key = _build_backend_model_key(backend_id, model)
        if key in self._llm_cache:
            return self._llm_cache[key]
        llm = self._llm_registry.get_llm(backend_id, model)
        self._llm_cache[key] = llm
        logger.info("query_engine_llm_cached backend_id=%s model=%s", backend_id, model)
        return llm


def _build_query(question: str, history: list[ConversationTurn]) -> str:
    if len(history) == 0:
        return question
    history_lines = [f"{turn.role}: {turn.message}" for turn in history]
    return f"Conversation history:\n{'\n'.join(history_lines)}\n\nQuestion:\n{question}"


def _to_query_result(response: Any) -> QueryResult:
    response_text = _extract_response_text(response)
    source_nodes = _extract_source_nodes(response)
    cleaned_text = _remove_inline_citations(response_text)
    if _is_effectively_empty_response_text(cleaned_text):
        cleaned_text = EMPTY_STREAM_RESPONSE_TEXT
        logger.warning("query_engine_answer_empty_response_text_replaced")
    return QueryResult(
        answer=cleaned_text,
        citations=[],
        grounded=len(source_nodes) > 0,
        retrieved_count=len(source_nodes),
    )


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return _require_non_empty(response, "response")
    if not hasattr(response, "response"):
        raise ValueError("query response must include response text")
    text = getattr(response, "response")
    if not isinstance(text, str):
        raise ValueError("query response text must be a string")
    return _require_non_empty(text, "response")


def _extract_source_nodes(response: Any) -> list[Any]:
    source_nodes = getattr(response, "source_nodes", [])
    if source_nodes is None:
        return []
    if not isinstance(source_nodes, list):
        raise ValueError("query response source_nodes must be a list")
    return source_nodes


def _remove_inline_citations(text: str) -> str:
    cleaned_text = INLINE_CITATION_PATTERN.sub("", text)
    return cleaned_text.strip()


def _is_effectively_empty_response_text(text: str) -> bool:
    normalized_text = text.strip().casefold()
    return normalized_text == "" or normalized_text in EMPTY_RESPONSE_SENTINELS


def _build_backend_model_key(backend_id: str, model: str) -> BackendModelKey:
    normalized_backend_id = _require_non_empty(backend_id, "backend_id")
    normalized_model = _require_non_empty(model, "model")
    return (normalized_backend_id, normalized_model)


def _validate_history(history: list[ConversationTurn]) -> None:
    for turn in history:
        if turn.role not in {"user", "assistant"}:
            raise ValueError("history role must be either 'user' or 'assistant'")
        _require_non_empty(turn.message, "history message")


def _require_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized


async def _iter_response_chunks(response: Any) -> AsyncIterator[str]:
    if not hasattr(response, "async_response_gen"):
        raise ValueError("streaming query response must provide async_response_gen")
    generator = response.async_response_gen()
    async for chunk in generator:
        if not isinstance(chunk, str):
            raise ValueError("streaming query chunk must be a string")
        yield chunk
