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


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    message: str


class ChatService:
    """Answer questions using document evidence plus model general knowledge."""

    def __init__(self, retrieval_service: RetrievalService, chat_client: ChatClient) -> None:
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client

    async def answer_question(self, question: str, history: list[ConversationTurn]) -> ChatResult:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        _validate_history(history)
        logger.info(
            "chat_answer_started question_length=%s history_turns=%s",
            len(question),
            len(history),
        )

        retrieval_query = _build_retrieval_query(question, history)
        retrieved_chunks = await self._retrieve_chunks_or_empty(retrieval_query)
        has_document_evidence = len(retrieved_chunks) > 0

        system_prompt = _build_system_prompt(has_document_evidence)
        user_prompt = _build_user_prompt(question, history, retrieved_chunks)
        answer = await self._chat_client.generate_chat_response(system_prompt, user_prompt)
        citations = [_chunk_to_citation(chunk) for chunk in retrieved_chunks]

        logger.info(
            "chat_answer_completed grounded=%s citations=%s history_turns=%s",
            has_document_evidence,
            len(citations),
            len(history),
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


def _validate_history(history: list[ConversationTurn]) -> None:
    for turn in history:
        if turn.role not in {"user", "assistant"}:
            raise ValueError("history role must be either 'user' or 'assistant'")
        if turn.message.strip() == "":
            raise ValueError("history message must not be empty")


def _build_retrieval_query(question: str, history: list[ConversationTurn]) -> str:
    recent_turns = history[-6:]
    if len(recent_turns) == 0:
        return question
    history_text = _format_history(recent_turns)
    return f"Conversation history:\n{history_text}\n\nCurrent question:\n{question}"


def _build_system_prompt(has_document_evidence: bool) -> str:
    if not has_document_evidence:
        document_section_instruction = (
            f"If the provided context does not contain evidence, state exactly: "
            f"'{NO_DOCUMENT_EVIDENCE}'. Then continue with a helpful answer using "
            "general knowledge."
        )
    else:
        document_section_instruction = (
            "When using document evidence, cite it inline using this format: "
            "[filename#chunk_id]."
        )

    return (
        "You are a conversational retrieval-augmented assistant. "
        "Use the conversation history to keep context across turns. "
        "Respond naturally as a normal conversation; do not force section headers "
        "or fixed templates unless the user explicitly asks for them. "
        f"{document_section_instruction} "
        "Never claim model general knowledge came from uploaded files."
    )


def _build_user_prompt(
    question: str, history: list[ConversationTurn], chunks: list[IndexedChunk]
) -> str:
    history_text = _format_history(history)
    context = _format_context(chunks)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )


def _format_history(history: list[ConversationTurn]) -> str:
    if len(history) == 0:
        return "[none]"
    return "\n".join([f"{turn.role}: {turn.message}" for turn in history])


def _format_context(chunks: list[IndexedChunk]) -> str:
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
    return "\n\n".join(chunk_blocks)


def _chunk_to_citation(chunk: IndexedChunk) -> dict[str, str | float | int | None]:
    citation = asdict(chunk)
    return {
        "doc_id": str(citation["doc_id"]),
        "filename": str(citation["filename"]),
        "chunk_id": str(citation["chunk_id"]),
        "score": float(citation["score"]),
        "page": citation["page"],
    }
