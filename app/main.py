"""FastAPI application entrypoint."""

from dataclasses import dataclass
import json
import logging
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

from app.config import Settings, load_environment_from_dotenv
from app.logging_config import configure_logging
from app.services.battleground import BattlegroundService, CompareStreamEvent
from app.services.chat import ChatService, ConversationTurn
from app.services.documents import DocumentService
from app.services.ingest import IngestService
from app.services.openrouter_client import OpenRouterClient
from app.services.parsers import (
    EmptyExtractionError,
    ParserDependencyError,
    UnsupportedFileTypeError,
)
from app.services.retrieval import RetrievalService
from app.services.vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="app/templates")


class ChatHistoryTurn(BaseModel):
    role: str
    message: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryTurn]


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    retrieved_count: int


class BattlegroundCompareRequest(BaseModel):
    message: str
    history: list[ChatHistoryTurn]
    model_a: str
    model_b: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "message")

    @field_validator("model_a")
    @classmethod
    def validate_model_a(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_a")

    @field_validator("model_b")
    @classmethod
    def validate_model_b(cls, value: str) -> str:
        return _require_non_empty_payload_value(value, "model_b")


class BattlegroundModelsResponse(BaseModel):
    models: list[str]


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
        model_a: str,
        model_b: str,
    ) -> AsyncIterator[CompareStreamEvent]:
        """Stream side-tagged compare events."""


@dataclass(frozen=True)
class AppServices:
    ingest_service: IngestService
    chat_service: ChatService
    document_service: DocumentService
    retrieval_service: RetrievalService
    chat_client: OpenRouterClient


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
    openrouter_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        embed_model=settings.openrouter_embed_model,
        chat_model=settings.openrouter_chat_model,
    )
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    retrieval_service = RetrievalService(
        embed_client=openrouter_client,
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
        min_relevance_score=settings.min_relevance_score,
    )
    ingest_service = IngestService(
        embed_client=openrouter_client,
        vector_store=vector_store,
        max_upload_mb=settings.max_upload_mb,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    document_service = DocumentService(vector_store=vector_store)
    chat_service = ChatService(
        retrieval_service=retrieval_service,
        chat_client=openrouter_client,
        document_service=document_service,
    )
    return AppServices(
        ingest_service=ingest_service,
        chat_service=chat_service,
        document_service=document_service,
        retrieval_service=retrieval_service,
        chat_client=openrouter_client,
    )


def _build_battleground_service(
    services: AppServices,
    settings: Settings,
) -> BattlegroundCompareService:
    return BattlegroundService(
        retrieval_service=services.retrieval_service,
        chat_client=services.chat_client,
        document_service=services.document_service,
        allowed_models=settings.openrouter_battleground_models,
    )


def _register_routes(app: FastAPI, services: AppServices, settings: Settings) -> None:
    _register_index_route(app)
    _register_health_route(app, settings)
    _register_upload_route(app, services)
    _register_documents_routes(app, services)
    _register_chat_routes(app, services)
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


def _register_chat_routes(app: FastAPI, services: AppServices) -> None:
    @app.post("/chat")
    async def chat(payload: ChatRequest) -> ChatResponse:
        logger.info(
            "chat_endpoint_called message_length=%s history_turns=%s",
            len(payload.message),
            len(payload.history),
        )
        history = _to_conversation_history(payload.history)
        try:
            result = await services.chat_service.answer_question(payload.message, history)
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
        logger.info(
            "chat_stream_endpoint_called message_length=%s history_turns=%s",
            len(payload.message),
            len(payload.history),
        )
        history = _to_conversation_history(payload.history)
        try:
            stream = await _resolve_chat_stream(
                services.chat_service.stream_answer_question(payload.message, history)
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
        logger.info(
            "battleground_models_list_requested model_count=%s",
            len(settings.openrouter_battleground_models),
        )
        return BattlegroundModelsResponse(models=list(settings.openrouter_battleground_models))

    @app.post("/battleground/compare/stream")
    async def battleground_compare_stream(payload: BattlegroundCompareRequest) -> StreamingResponse:
        logger.info(
            "battleground_compare_stream_endpoint_called message_length=%s history_turns=%s "
            "model_a=%s model_b=%s",
            len(payload.message),
            len(payload.history),
            payload.model_a,
            payload.model_b,
        )
        history = _to_conversation_history(payload.history)
        try:
            battleground_service = _build_battleground_service(services, settings)
            stream = await _prime_battleground_stream(
                battleground_service.compare_stream(
                    question=payload.message,
                    history=history,
                    model_a=payload.model_a,
                    model_b=payload.model_b,
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
