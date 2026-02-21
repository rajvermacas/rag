"""Grounded chat orchestration service."""

from dataclasses import asdict, dataclass
import logging
from typing import Protocol

from app.services.vector_store import IndexedChunk


logger = logging.getLogger(__name__)

UNKNOWN_ANSWER = "I do not know from the provided documents."


class RetrievalService(Protocol):
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        """Retrieve relevant chunks."""


class ChatClient(Protocol):
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a chat response from OpenRouter."""


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: list[dict[str, str | float | int | None]]
    grounded: bool
    retrieved_count: int


class ChatService:
    """Answer questions strictly using retrieved document context."""

    def __init__(self, retrieval_service: RetrievalService, chat_client: ChatClient) -> None:
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client

    async def answer_question(self, question: str) -> ChatResult:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        logger.info("chat_answer_started question_length=%s", len(question))

        try:
            retrieved_chunks = await self._retrieval_service.retrieve(question)
        except ValueError as exc:
            if str(exc) in {
                "retrieval returned no results",
                "no results passed relevance threshold",
            }:
                logger.info("chat_answer_unknown reason=%s", str(exc))
                return ChatResult(
                    answer=UNKNOWN_ANSWER,
                    citations=[],
                    grounded=False,
                    retrieved_count=0,
                )
            raise

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(question, retrieved_chunks)
        answer = await self._chat_client.generate_chat_response(system_prompt, user_prompt)
        citations = [_chunk_to_citation(chunk) for chunk in retrieved_chunks]

        logger.info(
            "chat_answer_completed grounded=true citations=%s",
            len(citations),
        )
        return ChatResult(
            answer=answer,
            citations=citations,
            grounded=True,
            retrieved_count=len(retrieved_chunks),
        )


def _build_system_prompt() -> str:
    return (
        "You are a document-grounded assistant. "
        "Answer only from the provided context chunks. "
        "If the context does not contain the answer, reply exactly: "
        "'I do not know from the provided documents.' "
        "Cite evidence using the provided chunk metadata."
    )


def _build_user_prompt(question: str, chunks: list[IndexedChunk]) -> str:
    chunk_blocks: list[str] = []
    for chunk in chunks:
        metadata = (
            f"doc_id={chunk.doc_id} "
            f"filename={chunk.filename} "
            f"chunk_id={chunk.chunk_id} "
            f"score={chunk.score:.4f} "
            f"page={chunk.page}"
        )
        chunk_blocks.append(f"[{metadata}]\n{chunk.text}")
    context = "\n\n".join(chunk_blocks)
    return f"Question:\n{question}\n\nContext:\n{context}"


def _chunk_to_citation(chunk: IndexedChunk) -> dict[str, str | float | int | None]:
    citation = asdict(chunk)
    return {
        "doc_id": str(citation["doc_id"]),
        "filename": str(citation["filename"]),
        "chunk_id": str(citation["chunk_id"]),
        "score": float(citation["score"]),
        "page": citation["page"],
    }
