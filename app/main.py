"""FastAPI application entrypoint."""

from dataclasses import dataclass
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import Settings, load_environment_from_dotenv
from app.logging_config import configure_logging
from app.services.chat import ChatService
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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    retrieved_count: int


@dataclass(frozen=True)
class AppServices:
    ingest_service: IngestService
    chat_service: ChatService


def create_app() -> FastAPI:
    dotenv_loaded = load_environment_from_dotenv(".env")
    settings = Settings.from_env()
    configure_logging(settings.app_log_level)
    logger.info("application_startup_dotenv_loaded loaded=%s", dotenv_loaded)
    services = _build_services(settings)
    application = FastAPI(title="RAG OpenRouter App")
    application.mount("/static", StaticFiles(directory="app/static"), name="static")
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
    chat_service = ChatService(
        retrieval_service=retrieval_service,
        chat_client=openrouter_client,
    )
    return AppServices(ingest_service=ingest_service, chat_service=chat_service)


def _register_routes(app: FastAPI, services: AppServices, settings: Settings) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        logger.info("index_page_requested")
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/health")
    async def health() -> dict[str, str]:
        logger.info("health_check_requested collection=%s", settings.chroma_collection_name)
        return {"status": "ok"}

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

    @app.post("/chat")
    async def chat(payload: ChatRequest) -> ChatResponse:
        logger.info("chat_endpoint_called message_length=%s", len(payload.message))
        try:
            result = await services.chat_service.answer_question(payload.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(
            answer=result.answer,
            citations=result.citations,
            grounded=result.grounded,
            retrieved_count=result.retrieved_count,
        )
