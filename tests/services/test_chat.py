import asyncio

from app.services.chat import ChatService, ConversationTurn, NO_DOCUMENT_EVIDENCE
from app.services.vector_store import IndexedChunk


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
        return (
            f"{NO_DOCUMENT_EVIDENCE} Revenue can refer to total income before expenses."
        )


class FakeChatClientWithEvidence:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if "[filename#chunk_id]" not in system_prompt:
            raise AssertionError("missing citation format requirement")
        if "exactly these two sections" in system_prompt:
            raise AssertionError("system prompt should not force rigid section output")
        if "a.txt" not in user_prompt:
            raise AssertionError("expected chunk metadata in context")
        if "Conversation history:\nuser: Earlier message" not in user_prompt:
            raise AssertionError("conversation history should be present in user prompt")
        return (
            "Revenue was 20. [a.txt#0] Revenue is commonly calculated before subtracting expenses."
        )


def test_chat_returns_unknown_without_evidence() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(service.answer_question("What is revenue?", history))

    assert result.grounded is False
    assert NO_DOCUMENT_EVIDENCE in result.answer
    assert result.citations == []
    assert result.retrieved_count == 0


def test_chat_returns_grounded_answer_with_citations() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatClientWithEvidence(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(service.answer_question("What is revenue?", history))

    assert result.grounded is True
    assert "[a.txt#0]" in result.answer
    assert result.retrieved_count == 1
    assert result.citations[0]["filename"] == "a.txt"


def test_chat_rejects_empty_history_message() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
    )
    history = [ConversationTurn(role="assistant", message=" ")]

    try:
        asyncio.run(service.answer_question("What is revenue?", history))
        raise AssertionError("expected ValueError for empty history message")
    except ValueError as exc:
        assert str(exc) == "history message must not be empty"
