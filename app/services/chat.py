"""Grounded chat orchestration service."""

from dataclasses import dataclass
import logging
import re
from typing import AsyncIterator, Protocol

from app.services.vector_store import IndexedChunk, IndexedDocument


logger = logging.getLogger(__name__)

NO_DOCUMENT_EVIDENCE = "No relevant evidence found in uploaded documents."
INLINE_CITATION_PATTERN = re.compile(
    r"\[[^\]\n]*(?:#chunk_id\s*=\s*\d+|#\d+)[^\]\n]*\]",
    re.IGNORECASE,
)
ANSWER_TOKEN_PATTERN = re.compile(r"\S+")
NUMBER_TOKEN_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?%?\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
MAX_CITATION_COUNT = 8
MIN_PHRASE_WORD_COUNT = 3
MIN_PHRASE_CHARACTER_COUNT = 12


class RetrievalService(Protocol):
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        """Retrieve relevant chunks."""


class ChatClient(Protocol):
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a chat response from OpenRouter."""

    async def stream_chat_response(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        """Stream a chat response from OpenRouter."""


class DocumentService(Protocol):
    def list_documents(self) -> list[IndexedDocument]:
        """Return all indexed documents."""


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


@dataclass(frozen=True)
class CitationCandidate:
    start: int
    end: int
    text: str
    doc_id: str
    filename: str
    chunk_id: str
    score: float


class ChatService:
    """Answer questions using document evidence plus model general knowledge."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        chat_client: ChatClient,
        document_service: DocumentService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chat_client = chat_client
        self._document_service = document_service

    async def answer_question(self, question: str, history: list[ConversationTurn]) -> ChatResult:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        _validate_history(history)
        documents = self._document_service.list_documents()
        logger.info(
            "chat_answer_started question_length=%s history_turns=%s available_documents=%s",
            len(question),
            len(history),
            len(documents),
        )
        if _is_document_inventory_question(question):
            answer = _build_document_inventory_answer(documents)
            logger.info(
                "chat_answer_completed_inventory_request document_count=%s",
                len(documents),
            )
            return ChatResult(
                answer=answer,
                citations=[],
                grounded=len(documents) > 0,
                retrieved_count=0,
            )

        retrieval_query = _build_retrieval_query(question, history)
        retrieved_chunks = await self._retrieve_chunks_or_empty(retrieval_query)
        has_document_evidence = len(retrieved_chunks) > 0

        system_prompt = _build_system_prompt(has_document_evidence)
        user_prompt = _build_user_prompt(question, history, retrieved_chunks, documents)
        raw_answer = await self._chat_client.generate_chat_response(system_prompt, user_prompt)
        answer = _remove_inline_citations(raw_answer)
        citations = _extract_answer_citations(answer, retrieved_chunks)

        logger.info(
            "chat_answer_completed grounded=%s retrieved_count=%s history_turns=%s citation_count=%s",
            has_document_evidence,
            len(retrieved_chunks),
            len(history),
            len(citations),
        )
        return ChatResult(
            answer=answer,
            citations=citations,
            grounded=has_document_evidence,
            retrieved_count=len(retrieved_chunks),
        )

    async def stream_answer_question(
        self, question: str, history: list[ConversationTurn]
    ) -> AsyncIterator[str]:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        _validate_history(history)
        documents = self._document_service.list_documents()
        logger.info(
            "chat_stream_started question_length=%s history_turns=%s available_documents=%s",
            len(question),
            len(history),
            len(documents),
        )
        if _is_document_inventory_question(question):
            answer = _build_document_inventory_answer(documents)
            logger.info("chat_stream_completed_inventory_request document_count=%s", len(documents))
            return _single_chunk_stream(answer)

        retrieval_query = _build_retrieval_query(question, history)
        retrieved_chunks = await self._retrieve_chunks_or_empty(retrieval_query)
        has_document_evidence = len(retrieved_chunks) > 0
        system_prompt = _build_system_prompt(has_document_evidence)
        user_prompt = _build_user_prompt(question, history, retrieved_chunks, documents)
        return self._chat_client.stream_chat_response(system_prompt, user_prompt)

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
            "general knowledge. Exception: if the user asks which uploaded documents "
            "are available, answer directly from the 'Available uploaded documents' "
            "section and do not output the no-evidence sentence. Do not include "
            "citations, bracketed references, or chunk identifiers in your answer."
        )
    else:
        document_section_instruction = (
            "Use document evidence, but do not include citations, bracketed references, "
            "or chunk identifiers in your answer."
        )

    return (
        "You are a conversational retrieval-augmented assistant. "
        "Use the conversation history to keep context across turns. "
        "Respond naturally as a normal conversation; do not force section headers "
        "or fixed templates unless the user explicitly asks for them. "
        "The prompt includes an 'Available uploaded documents' section; when the user "
        "asks what documents are available, answer with exact filenames from that list. "
        f"{document_section_instruction} "
        "Never claim model general knowledge came from uploaded files."
    )


def _build_user_prompt(
    question: str,
    history: list[ConversationTurn],
    chunks: list[IndexedChunk],
    documents: list[IndexedDocument],
) -> str:
    history_text = _format_history(history)
    context = _format_context(chunks)
    uploaded_documents = _format_uploaded_documents(documents)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Available uploaded documents:\n{uploaded_documents}\n\n"
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


def _format_uploaded_documents(documents: list[IndexedDocument]) -> str:
    if len(documents) == 0:
        return "[none]"
    return "\n".join(
        [
            f"{index + 1}. {document.filename} (doc_id={document.doc_id}, "
            f"chunks={document.chunks_indexed})"
            for index, document in enumerate(documents)
        ]
    )


async def _single_chunk_stream(answer: str) -> AsyncIterator[str]:
    yield answer


def _remove_inline_citations(text: str) -> str:
    cleaned_text = INLINE_CITATION_PATTERN.sub("", text)
    return cleaned_text.strip()


def _extract_answer_citations(
    answer: str, chunks: list[IndexedChunk]
) -> list[dict[str, str | float | int | None]]:
    if answer.strip() == "":
        raise ValueError("answer must not be empty")
    if len(chunks) == 0:
        return []

    candidates: list[CitationCandidate] = []
    for chunk in chunks:
        normalized_chunk = _normalize_whitespace(chunk.text).lower()
        if normalized_chunk == "":
            raise ValueError(
                f"retrieved chunk text must not be empty for doc_id={chunk.doc_id} "
                f"chunk_id={chunk.chunk_id}"
            )
        candidates.extend(_find_phrase_candidates(answer, normalized_chunk, chunk))
        candidates.extend(_find_numeric_candidates(answer, normalized_chunk, chunk))

    selected_candidates = _select_best_candidates(candidates)
    logger.info(
        "chat_citation_extraction_completed candidates=%s selected=%s",
        len(candidates),
        len(selected_candidates),
    )
    return [_serialize_citation(candidate) for candidate in selected_candidates]


def _find_phrase_candidates(
    answer: str, normalized_chunk: str, chunk: IndexedChunk
) -> list[CitationCandidate]:
    token_matches = list(ANSWER_TOKEN_PATTERN.finditer(answer))
    if len(token_matches) < MIN_PHRASE_WORD_COUNT:
        return []

    max_window_size = min(12, len(token_matches))
    phrase_candidates: list[CitationCandidate] = []
    for window_size in range(max_window_size, MIN_PHRASE_WORD_COUNT - 1, -1):
        for start_index in range(0, len(token_matches) - window_size + 1):
            end_index = start_index + window_size - 1
            start = token_matches[start_index].start()
            end = token_matches[end_index].end()
            phrase = answer[start:end]
            normalized_phrase = _normalize_whitespace(phrase).lower()
            if len(normalized_phrase) < MIN_PHRASE_CHARACTER_COUNT:
                continue
            if normalized_phrase not in normalized_chunk:
                continue
            phrase_candidates.append(_build_candidate(start, end, phrase, chunk))
    return phrase_candidates


def _find_numeric_candidates(
    answer: str, normalized_chunk: str, chunk: IndexedChunk
) -> list[CitationCandidate]:
    numeric_candidates: list[CitationCandidate] = []
    for token_match in NUMBER_TOKEN_PATTERN.finditer(answer):
        number_token = token_match.group(0)
        if number_token not in normalized_chunk:
            continue
        numeric_candidates.append(
            _build_candidate(token_match.start(), token_match.end(), number_token, chunk)
        )
    return numeric_candidates


def _build_candidate(start: int, end: int, text: str, chunk: IndexedChunk) -> CitationCandidate:
    if start >= end:
        raise ValueError("citation candidate start must be less than end")
    cleaned_text = text.strip()
    if cleaned_text == "":
        raise ValueError("citation candidate text must not be empty")
    return CitationCandidate(
        start=start,
        end=end,
        text=cleaned_text,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
    )


def _select_best_candidates(candidates: list[CitationCandidate]) -> list[CitationCandidate]:
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -(candidate.end - candidate.start),
            -candidate.score,
            candidate.start,
        ),
    )
    selected_candidates: list[CitationCandidate] = []
    occupied_ranges: list[tuple[int, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in ranked_candidates:
        candidate_key = (candidate.text.lower(), candidate.filename)
        if candidate_key in seen_pairs:
            continue
        if _range_overlaps(candidate.start, candidate.end, occupied_ranges):
            continue
        selected_candidates.append(candidate)
        occupied_ranges.append((candidate.start, candidate.end))
        seen_pairs.add(candidate_key)
        if len(selected_candidates) >= MAX_CITATION_COUNT:
            break
    return sorted(selected_candidates, key=lambda candidate: candidate.start)


def _range_overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    for existing_start, existing_end in ranges:
        if start < existing_end and end > existing_start:
            return True
    return False


def _serialize_citation(candidate: CitationCandidate) -> dict[str, str | float | int | None]:
    return {
        "text": candidate.text,
        "filename": candidate.filename,
        "doc_id": candidate.doc_id,
        "chunk_id": candidate.chunk_id,
        "score": candidate.score,
    }


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _is_document_inventory_question(question: str) -> bool:
    normalized_question = question.lower()
    inventory_terms = [
        "what documents",
        "which documents",
        "list documents",
        "uploaded documents",
        "uploaded files",
        "files you have access",
        "documents you have access",
        "what files",
    ]
    return any(term in normalized_question for term in inventory_terms)


def _build_document_inventory_answer(documents: list[IndexedDocument]) -> str:
    if len(documents) == 0:
        return "I currently have no uploaded documents indexed."
    lines = ["I currently have access to these uploaded documents:"]
    for index, document in enumerate(documents):
        lines.append(
            f"{index + 1}. {document.filename} (doc_id: {document.doc_id}, "
            f"chunks: {document.chunks_indexed})"
        )
    return "\n".join(lines)
