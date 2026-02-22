"""FastAPI application entrypoint."""

from dataclasses import dataclass
import json
import logging
import warnings
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import (
    request_validation_exception_handler as fastapi_request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from pydantic.warnings import UnsupportedFieldAttributeWarning

from app.constants import OPENROUTER_API_BASE_URL
from app.config import Settings, load_environment_from_dotenv
from app.logging_config import configure_logging
from app.services.battleground import BattlegroundService, CompareStreamEvent
from app.services.chat import ChatService, ConversationTurn
from app.services.chat_provider_models import ChatModelOption
from app.services.chat_provider_router import ChatProviderRouter
from app.services.documents import DocumentService
from app.services.indexing import IndexingService
from app.services.llm_registry import LLMRegistry
from app.services.query_engine import QueryEngineService
from app.services.parsers import (
    EmptyExtractionError,
    ParserDependencyError,
    UnsupportedFileTypeError,
)
from app.startup_warmup import build_backend_model_pairs, warm_up_runtime_dependencies


logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="app/templates")


class ChatHistoryTurn(BaseModel):
    role: str
    message: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryTurn]
    backend_id: str
    model: str

    @field_validator("backend_id")
    @classmethod
    def validate_backend_id(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "backend_id")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model")


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    retrieved_count: int


class BattlegroundCompareRequest(BaseModel):
    message: str
    history: list[ChatHistoryTurn]
    model_a_backend_id: str
    model_a: str
    model_b_backend_id: str
    model_b: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "message")

    @field_validator("model_a_backend_id")
    @classmethod
    def validate_model_a_backend_id(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_a_backend_id")

    @field_validator("model_a")
    @classmethod
    def validate_model_a(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_a")

    @field_validator("model_b_backend_id")
    @classmethod
    def validate_model_b_backend_id(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_b_backend_id")

    @field_validator("model_b")
    @classmethod
    def validate_model_b(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_b")


class ChatModelOptionResponse(BaseModel):
    backend_id: str
    provider: str
    model: str
    label: str


class BattlegroundModelsResponse(BaseModel):
    models: list[ChatModelOptionResponse]


class ChatModelsResponse(BaseModel):
    models: list[ChatModelOptionResponse]


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


class BattlegroundCompareService(Protocol):
    def compare_stream(
        self,
        question: str,
        history: list[ConversationTurn],
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ) -> AsyncIterator[CompareStreamEvent]:
        """Stream side-tagged compare events."""


class IngestServiceProtocol(Protocol):
    async def ingest_upload(self, upload: UploadFile) -> Any:
        """Ingest uploaded file content."""


class ChatServiceProtocol(Protocol):
    async def answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> Any:
        """Answer question non-streaming."""

    async def stream_answer_question(
        self,
        question: str,
        history: list[ConversationTurn],
        backend_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Answer question with streaming output."""


class DocumentServiceProtocol(Protocol):
    def list_documents(self) -> list[Any]:
        """List indexed documents."""

    def delete_document(self, doc_id: str) -> int:
        """Delete indexed document by id."""


@dataclass(frozen=True)
class AppServices:
    ingest_service: IngestServiceProtocol
    chat_service: ChatServiceProtocol
    document_service: DocumentServiceProtocol
    retrieval_service: Any
    chat_provider_router: ChatProviderRouter


def create_app() -> FastAPI:
    dotenv_loaded = load_environment_from_dotenv(".env")
    settings = Settings.from_env()
    configure_logging(settings.app_log_level)
    logger.info("application_startup_dotenv_loaded loaded=%s", dotenv_loaded)
    services = _build_services(settings)
    application = FastAPI(title="RAG OpenRouter App")
    application.mount("/static", StaticFiles(directory="app/static"), name="static")
    _register_request_validation_handlers(application)
    _register_routes(application, services, settings)
    return application


def _build_services(settings: Settings) -> AppServices:
    collection = _build_chroma_collection(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    embed_model = _build_openrouter_embedding_model(
        api_key=settings.openrouter_api_key,
        embed_model=settings.openrouter_embed_model,
    )
    indexing_service = IndexingService(
        collection=collection,
        max_upload_mb=settings.max_upload_mb,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embed_model=embed_model,
    )
    document_service = DocumentService(vector_store=indexing_service)
    llm_registry = LLMRegistry(settings.chat_backend_profiles)
    query_service = QueryEngineService(
        llm_registry=llm_registry,
        engine_factory=_LlamaIndexQueryEngineFactory(
            collection=collection,
            embed_model=embed_model,
            top_k=settings.retrieval_top_k,
        ),
    )
    backend_model_pairs = build_backend_model_pairs(settings.chat_backend_profiles)
    warm_up_runtime_dependencies(query_service, backend_model_pairs)
    chat_service = ChatService(query_service=query_service)
    chat_provider_router = _build_chat_provider_router(settings)
    return AppServices(
        ingest_service=indexing_service,
        chat_service=chat_service,
        document_service=document_service,
        retrieval_service=query_service,
        chat_provider_router=chat_provider_router,
    )


def _build_battleground_service(
    services: AppServices,
) -> BattlegroundCompareService:
    return BattlegroundService(query_service=services.chat_service)


def _build_chat_provider_router(settings: Settings) -> ChatProviderRouter:
    providers: dict[str, _UnusedBackendChatProvider] = {}
    for backend_id, profile in settings.chat_backend_profiles.items():
        providers[backend_id] = _UnusedBackendChatProvider(backend_id, profile.provider)
    logger.info("chat_provider_router_built backend_count=%s", len(providers))
    return ChatProviderRouter(
        backend_profiles=settings.chat_backend_profiles,
        providers=providers,
    )


class _UnusedBackendChatProvider:
    """Compatibility provider for model-option routing during migration."""

    def __init__(self, backend_id: str, provider: str) -> None:
        self._backend_id = backend_id
        self._provider = provider

    async def generate_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise RuntimeError(
            "chat provider router transport is deprecated and must not be called: "
            f"backend_id={self._backend_id} provider={self._provider} model={model}"
        )

    async def stream_chat_response_with_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        raise RuntimeError(
            "chat provider router stream transport is deprecated and must not be called: "
            f"backend_id={self._backend_id} provider={self._provider} model={model}"
        )
        yield ""


class _LlamaIndexQueryEngineFactory:
    def __init__(self, collection: Any, embed_model: Any, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if collection is None:
            raise ValueError("collection must not be None")
        if embed_model is None:
            raise ValueError("embed_model must not be None")
        self._collection = collection
        self._embed_model = embed_model
        self._top_k = top_k

    def build_query_engine(self, llm: Any) -> Any:
        return _build_llamaindex_query_engine(
            collection=self._collection,
            embed_model=self._embed_model,
            llm=llm,
            top_k=self._top_k,
            streaming=False,
        )

    def build_streaming_query_engine(self, llm: Any) -> Any:
        return _build_llamaindex_query_engine(
            collection=self._collection,
            embed_model=self._embed_model,
            llm=llm,
            top_k=self._top_k,
            streaming=True,
        )

    def has_indexed_documents(self) -> bool:
        document_count = self._collection.count()
        if not isinstance(document_count, int) or document_count < 0:
            raise ValueError("chroma collection count must be a non-negative integer")
        logger.info("query_engine_factory_documents_count count=%s", document_count)
        return document_count > 0


def _ignore_known_llamaindex_validate_default_warning() -> None:
    logger.debug("suppressing_known_llamaindex_validate_default_warning")
    warnings.filterwarnings(
        "ignore",
        message=(
            r"The 'validate_default' attribute with value True was provided to the "
            r"`Field\(\)` function, which has no effect in the context it was used\..*"
        ),
        category=UnsupportedFieldAttributeWarning,
    )


def _build_chroma_collection(persist_dir: str, collection_name: str) -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for app wiring: chromadb") from exc
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name)


def _build_openrouter_embedding_model(api_key: str, embed_model: str) -> Any:
    try:
        with warnings.catch_warnings():
            _ignore_known_llamaindex_validate_default_warning()
            from llama_index.embeddings.openai import OpenAIEmbedding
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency for app wiring: llama-index-embeddings-openai"
        ) from exc
    return OpenAIEmbedding(
        model=embed_model,
        api_key=api_key,
        api_base=OPENROUTER_API_BASE_URL,
    )


def _build_llamaindex_query_engine(
    collection: Any,
    embed_model: Any,
    llm: Any,
    top_k: int,
    streaming: bool,
) -> Any:
    try:
        with warnings.catch_warnings():
            _ignore_known_llamaindex_validate_default_warning()
            from llama_index.core import VectorStoreIndex
            from llama_index.vector_stores.chroma import ChromaVectorStore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for app wiring: llama-index") from exc
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        streaming=streaming,
    )


def _register_routes(app: FastAPI, services: AppServices, settings: Settings) -> None:
    _register_index_route(app)
    _register_health_route(app, settings)
    _register_upload_route(app, services)
    _register_documents_routes(app, services)
    _register_chat_routes(app, services, settings)
    _register_battleground_routes(app, services, settings)


def _register_request_validation_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        if not _is_battleground_compare_stream_request(request):
            return await fastapi_request_validation_exception_handler(request, exc)
        detail = _build_battleground_validation_detail(exc.errors())
        logger.info(
            "battleground_compare_stream_payload_validation_failed detail=%s",
            detail,
        )
        return JSONResponse(status_code=400, content={"detail": detail})


def _is_battleground_compare_stream_request(request: Request) -> bool:
    return request.method.upper() == "POST" and request.url.path == "/battleground/compare/stream"


def _build_battleground_validation_detail(errors: list[dict[str, Any]]) -> str:
    messages = [_build_battleground_validation_message(error) for error in errors]
    return f"invalid battleground compare payload: {'; '.join(messages)}"


def _build_battleground_validation_message(error: dict[str, Any]) -> str:
    if _is_non_object_body_payload_error(error):
        return "payload must be a JSON object"
    if _is_malformed_json_payload_error(error):
        return "payload must be valid JSON"
    field_name = _extract_validation_field_name(error["loc"])
    if error["type"] == "missing":
        return f"{field_name} is required"
    message = _strip_pydantic_value_error_prefix(str(error["msg"]))
    if message.startswith(f"{field_name} "):
        return message
    return f"{field_name}: {message}"


def _is_non_object_body_payload_error(error: dict[str, Any]) -> bool:
    location = tuple(error["loc"])
    return location == ("body",) and error["type"] in {
        "dict_type",
        "model_attributes_type",
    }


def _is_malformed_json_payload_error(error: dict[str, Any]) -> bool:
    location = tuple(error["loc"])
    return len(location) > 0 and location[0] == "body" and error["type"] == "json_invalid"


def _extract_validation_field_name(location: tuple[Any, ...]) -> str:
    field_path = [str(part) for part in location if part != "body"]
    if len(field_path) == 0:
        raise ValueError("request validation location did not include a body field")
    return ".".join(field_path)


def _strip_pydantic_value_error_prefix(message: str) -> str:
    prefix = "Value error, "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message


def _require_non_empty_payload_value(value: str, field_name: str) -> str:
    if value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    return value


def _register_index_route(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        logger.info("index_page_requested")
        return templates.TemplateResponse(request=request, name="index.html")


def _register_health_route(app: FastAPI, settings: Settings) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        logger.info("health_check_requested collection=%s", settings.chroma_collection_name)
        return {"status": "ok"}


def _register_upload_route(app: FastAPI, services: AppServices) -> None:
    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        logger.info(
            "upload_endpoint_called filename=%s content_type=%s",
            file.filename,
            file.content_type,
        )
        try:
            result = await services.ingest_service.ingest_upload(file)
        except (
            ValueError,
            UnsupportedFileTypeError,
            EmptyExtractionError,
            ParserDependencyError,
        ) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"doc_id": result.doc_id, "chunks_indexed": result.chunks_indexed}


def _register_documents_routes(app: FastAPI, services: AppServices) -> None:
    @app.get("/documents")
    async def list_documents() -> DocumentListResponse:
        logger.info("list_documents_endpoint_called")
        documents = services.document_service.list_documents()
        return DocumentListResponse(
            documents=[
                DocumentSummaryResponse(
                    doc_id=document.doc_id,
                    filename=document.filename,
                    chunks_indexed=document.chunks_indexed,
                )
                for document in documents
            ]
        )

    @app.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> DeleteDocumentResponse:
        logger.info("delete_document_endpoint_called doc_id=%s", doc_id)
        try:
            chunks_deleted = services.document_service.delete_document(doc_id)
        except ValueError as exc:
            error_message = str(exc)
            if "not found" in error_message:
                raise HTTPException(status_code=404, detail=error_message) from exc
            raise HTTPException(status_code=400, detail=error_message) from exc
        return DeleteDocumentResponse(doc_id=doc_id, chunks_deleted=chunks_deleted)


def _register_chat_routes(
    app: FastAPI,
    services: AppServices,
    settings: Settings,
) -> None:
    @app.get("/models/chat")
    async def list_chat_models() -> ChatModelsResponse:
        model_options = services.chat_provider_router.list_model_options()
        logger.info(
            "chat_models_list_requested model_count=%s",
            len(model_options),
        )
        return ChatModelsResponse(
            models=_to_chat_model_option_responses(model_options),
        )

    @app.post("/chat")
    async def chat(payload: ChatRequest) -> ChatResponse:
        try:
            selected_option = _require_allowed_chat_model_option(
                backend_id=payload.backend_id,
                model=payload.model,
                model_options=services.chat_provider_router.list_model_options(),
            )
            logger.info(
                "chat_endpoint_called message_length=%s history_turns=%s backend_id=%s "
                "provider=%s model=%s",
                len(payload.message),
                len(payload.history),
                selected_option.backend_id,
                selected_option.provider,
                selected_option.model,
            )
            history = _to_conversation_history(payload.history)
            result = await services.chat_service.answer_question(
                question=payload.message,
                history=history,
                backend_id=selected_option.backend_id,
                model=selected_option.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(
            answer=result.answer,
            citations=[],
            grounded=result.grounded,
            retrieved_count=result.retrieved_count,
        )

    @app.post("/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        try:
            selected_option = _require_allowed_chat_model_option(
                backend_id=payload.backend_id,
                model=payload.model,
                model_options=services.chat_provider_router.list_model_options(),
            )
            logger.info(
                "chat_stream_endpoint_called message_length=%s history_turns=%s "
                "backend_id=%s provider=%s model=%s",
                len(payload.message),
                len(payload.history),
                selected_option.backend_id,
                selected_option.provider,
                selected_option.model,
            )
            history = _to_conversation_history(payload.history)
            stream = await _resolve_chat_stream(
                services.chat_service.stream_answer_question(
                    question=payload.message,
                    history=history,
                    backend_id=selected_option.backend_id,
                    model=selected_option.model,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(_stream_chat_chunks(stream), media_type="text/plain; charset=utf-8")


def _register_battleground_routes(
    app: FastAPI,
    services: AppServices,
    settings: Settings,
) -> None:
    @app.get("/models/battleground")
    async def list_battleground_models() -> BattlegroundModelsResponse:
        model_options = services.chat_provider_router.list_model_options()
        logger.info(
            "battleground_models_list_requested model_count=%s",
            len(model_options),
        )
        return BattlegroundModelsResponse(
            models=_to_chat_model_option_responses(model_options),
        )

    @app.post("/battleground/compare/stream")
    async def battleground_compare_stream(payload: BattlegroundCompareRequest) -> StreamingResponse:
        model_options = services.chat_provider_router.list_model_options()
        logger.info(
            "battleground_compare_stream_endpoint_called message_length=%s history_turns=%s "
            "model_a_backend_id=%s model_a=%s model_b_backend_id=%s model_b=%s",
            len(payload.message),
            len(payload.history),
            payload.model_a_backend_id,
            payload.model_a,
            payload.model_b_backend_id,
            payload.model_b,
        )
        history = _to_conversation_history(payload.history)
        try:
            selected_option_a = _require_allowed_chat_model_option(
                backend_id=payload.model_a_backend_id,
                model=payload.model_a,
                model_options=model_options,
            )
            selected_option_b = _require_allowed_chat_model_option(
                backend_id=payload.model_b_backend_id,
                model=payload.model_b,
                model_options=model_options,
            )
            _validate_distinct_battleground_model_choices(
                selected_option_a,
                selected_option_b,
            )
            battleground_service = _build_battleground_service(services)
            stream = await _prime_battleground_stream(
                battleground_service.compare_stream(
                    question=payload.message,
                    history=history,
                    model_a_backend_id=selected_option_a.backend_id,
                    model_a=selected_option_a.model,
                    model_b_backend_id=selected_option_b.backend_id,
                    model_b=selected_option_b.model,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_battleground_events(stream),
            media_type="application/x-ndjson",
        )


def _to_conversation_history(history: list[ChatHistoryTurn]) -> list[ConversationTurn]:
    return [ConversationTurn(role=turn.role, message=turn.message) for turn in history]


def _to_chat_model_option_responses(
    model_options: tuple[ChatModelOption, ...],
) -> list[ChatModelOptionResponse]:
    return [
        ChatModelOptionResponse(
            backend_id=option.backend_id,
            provider=option.provider,
            model=option.model,
            label=option.label,
        )
        for option in model_options
    ]


def _require_allowed_chat_model_option(
    backend_id: str,
    model: str,
    model_options: tuple[ChatModelOption, ...],
) -> ChatModelOption:
    normalized_backend_id = backend_id.strip()
    normalized_model = model.strip()
    if normalized_backend_id == "":
        raise ValueError("backend_id must not be empty")
    if normalized_model == "":
        raise ValueError("model must not be empty")
    backend_matches = [
        option for option in model_options if option.backend_id == normalized_backend_id
    ]
    if len(backend_matches) == 0:
        raise ValueError("backend_id is not allowed")
    for option in backend_matches:
        if option.model == normalized_model:
            return option
    raise ValueError("model is not allowed for backend_id")

def _validate_distinct_battleground_model_choices(
    model_option_a: ChatModelOption,
    model_option_b: ChatModelOption,
) -> None:
    if (
        model_option_a.backend_id == model_option_b.backend_id
        and model_option_a.model == model_option_b.model
    ):
        raise ValueError("model_a and model_b must be different")

async def _stream_chat_chunks(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    chunk_count = 0
    async for chunk in stream:
        if chunk == "":
            continue
        chunk_count += 1
        yield chunk
    logger.info("chat_stream_completed chunk_count=%s", chunk_count)

async def _resolve_chat_stream(stream_or_awaitable: Any) -> AsyncIterator[str]:
    if hasattr(stream_or_awaitable, "__aiter__"):
        return stream_or_awaitable
    if hasattr(stream_or_awaitable, "__await__"):
        resolved_stream = await stream_or_awaitable
        if not hasattr(resolved_stream, "__aiter__"):
            raise ValueError("chat stream resolver expected an async iterator")
        return resolved_stream
    raise ValueError("chat stream resolver expected an awaitable or async iterator")

async def _prime_battleground_stream(
    stream: AsyncIterator[CompareStreamEvent],
) -> AsyncIterator[CompareStreamEvent]:
    try:
        first_event = await anext(stream)
    except StopAsyncIteration as exc:
        raise RuntimeError("battleground stream produced no events") from exc
    return _prepend_battleground_event(first_event, stream)


async def _prepend_battleground_event(
    first_event: CompareStreamEvent,
    stream: AsyncIterator[CompareStreamEvent],
) -> AsyncIterator[CompareStreamEvent]:
    yield first_event
    async for event in stream:
        yield event


async def _stream_battleground_events(
    stream: AsyncIterator[CompareStreamEvent],
) -> AsyncIterator[str]:
    event_count = 0
    async for event in stream:
        event_count += 1
        yield _serialize_battleground_event(event)
    logger.info("battleground_compare_stream_completed event_count=%s", event_count)


def _serialize_battleground_event(event: CompareStreamEvent) -> str:
    side = _validate_battleground_event_side(event.side)
    payload: dict[str, Any] = {"side": side}
    if event.kind == "chunk":
        if event.chunk is None or event.chunk == "":
            raise ValueError("battleground chunk event must include chunk")
        payload["chunk"] = event.chunk
    elif event.kind == "done":
        payload["done"] = True
    elif event.kind == "error":
        if event.error is None or event.error.strip() == "":
            raise ValueError("battleground error event must include error")
        payload["error"] = event.error
    else:
        raise ValueError(f"unsupported battleground event kind: {event.kind}")
    return f"{json.dumps(payload)}\n"


def _validate_battleground_event_side(side: Any) -> str:
    if not isinstance(side, str):
        raise ValueError("battleground event side must be a string")
    if side.strip() == "":
        raise ValueError("battleground event side must not be empty")
    return side
