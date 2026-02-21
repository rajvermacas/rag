"""Grounded chat orchestration service."""

from dataclasses import asdict, dataclass
import logging
from typing import Protocol

from app.services.vector_store import IndexedChunk


logger = logging.getLogger(__name__)

NO_DOCUMENT_EVIDENCE = "No relevant evidence found in uploaded documents."


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
    """Answer questions using document evidence plus model general knowledge."""

    def __init__(self, retrieval_service: RetrievalService, chat_client: ChatClient) -> None:
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client

    async def answer_question(self, question: str) -> ChatResult:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        logger.info("chat_answer_started question_length=%s", len(question))

        retrieved_chunks = await self._retrieve_chunks_or_empty(question)
        has_document_evidence = len(retrieved_chunks) > 0

        system_prompt = _build_system_prompt(has_document_evidence)
        user_prompt = _build_user_prompt(question, retrieved_chunks)
        answer = await self._chat_client.generate_chat_response(system_prompt, user_prompt)
        citations = [_chunk_to_citation(chunk) for chunk in retrieved_chunks]

        logger.info(
            "chat_answer_completed grounded=%s citations=%s",
            has_document_evidence,
            len(citations),
        )
        return ChatResult(
            answer=answer,
            citations=citations,
            grounded=has_document_evidence,
            retrieved_count=len(retrieved_chunks),
        )

    async def _retrieve_chunks_or_empty(self, question: str) -> list[IndexedChunk]:
        try:
            return await self._retrieval_service.retrieve(question)
        except ValueError as exc:
            if str(exc) not in {
                "retrieval returned no results",
                "no results passed relevance threshold",
            }:
                raise
            logger.info("chat_retrieval_no_evidence reason=%s", str(exc))
            return []


def _build_system_prompt(has_document_evidence: bool) -> str:
    if not has_document_evidence:
        document_section_instruction = (
            "The document section must be exactly: "
            f"'{NO_DOCUMENT_EVIDENCE}'."
        )
    else:
        document_section_instruction = (
            "In the document section, use only the provided context and add inline "
            "citations in this format: [filename#chunk_id]."
        )

    return (
        "You are a retrieval-augmented assistant with two responsibilities. "
        "Always answer using exactly these two sections in order:\n"
        "1) From uploaded documents (with citations):\n"
        "2) From general knowledge (not from uploaded documents):\n"
        f"{document_section_instruction} "
        "In the general knowledge section, clearly separate model knowledge from "
        "document evidence and never imply it came from uploaded files."
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
