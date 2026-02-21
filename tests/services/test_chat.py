import asyncio

from app.services.chat import ChatService, ConversationTurn, NO_DOCUMENT_EVIDENCE
from app.services.vector_store import IndexedChunk, IndexedDocument


class FakeRetrievalNoEvidence:
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        raise ValueError("no results passed relevance threshold")


class FakeRetrievalWithEvidence:
    async def retrieve(self, question: str) -> list[IndexedChunk]:
        return [
            IndexedChunk(
                doc_id="doc-1",
                filename="a.txt",
                chunk_id="0",
                text="The document says revenue was 20.",
                score=0.92,
                page=None,
            )
        ]


class FakeDocumentService:
    def list_documents(self):
        return [
            IndexedDocument(doc_id="doc-1", filename="a.txt", chunks_indexed=1),
            IndexedDocument(doc_id="doc-2", filename="b.pdf", chunks_indexed=3),
        ]


class FakeChatClientNoEvidence:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if NO_DOCUMENT_EVIDENCE not in system_prompt:
            raise AssertionError("missing explicit no-evidence guidance")
        if "exactly these two sections" in system_prompt:
            raise AssertionError("system prompt should not force rigid section output")
        if "Context:\n" not in user_prompt:
            raise AssertionError("context section must be present")
        if "Conversation history:\nuser: Earlier message" not in user_prompt:
            raise AssertionError("conversation history should be present in user prompt")
        if "Available uploaded documents:" not in user_prompt:
            raise AssertionError("uploaded document catalog should be present")
        if "a.txt" not in user_prompt or "b.pdf" not in user_prompt:
            raise AssertionError("document filenames should be present in user prompt")
        return f"{NO_DOCUMENT_EVIDENCE} Revenue can refer to total income before expenses."

    async def stream_chat_response(self, system_prompt: str, user_prompt: str):
        yield await self.generate_chat_response(system_prompt, user_prompt)


class FakeChatClientWithEvidence:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if "do not include citations" not in system_prompt:
            raise AssertionError("missing no-citation requirement")
        if "exactly these two sections" in system_prompt:
            raise AssertionError("system prompt should not force rigid section output")
        if "a.txt" not in user_prompt:
            raise AssertionError("expected chunk metadata in context")
        if "Available uploaded documents:" not in user_prompt:
            raise AssertionError("uploaded document catalog should be present")
        if "Conversation history:\nuser: Earlier message" not in user_prompt:
            raise AssertionError("conversation history should be present in user prompt")
        return "Revenue was 20. [a.txt#0] Revenue is commonly calculated before subtracting expenses."

    async def stream_chat_response(self, system_prompt: str, user_prompt: str):
        yield "Revenue was "
        yield "20. [a.txt#0]"


class FakeChatClientNotExpected:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("chat client should not be called for inventory questions")

    async def stream_chat_response(self, system_prompt: str, user_prompt: str):
        raise AssertionError("chat client should not be called for inventory questions")


def test_chat_returns_unknown_without_evidence() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(service.answer_question("What is revenue?", history))

    assert result.grounded is False
    assert NO_DOCUMENT_EVIDENCE in result.answer
    assert result.citations == []
    assert result.retrieved_count == 0


def test_chat_returns_grounded_answer_without_citations() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatClientWithEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(service.answer_question("What is revenue?", history))

    assert result.grounded is True
    assert "[a.txt#0]" not in result.answer
    assert result.retrieved_count == 1
    assert result.citations == []


def test_chat_stream_returns_chunks() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatClientWithEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    async def collect_chunks() -> list[str]:
        stream = await service.stream_answer_question("What is revenue?", history)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect_chunks())
    assert len(chunks) == 2


def test_chat_rejects_empty_history_message() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="assistant", message=" ")]

    try:
        asyncio.run(service.answer_question("What is revenue?", history))
        raise AssertionError("expected ValueError for empty history message")
    except ValueError as exc:
        assert str(exc) == "history message must not be empty"


def test_chat_answers_document_inventory_without_model_call() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNotExpected(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(service.answer_question("What documents do you have access to?", history))

    assert result.grounded is True
    assert result.retrieved_count == 0
    assert "a.txt" in result.answer
    assert "b.pdf" in result.answer
