import asyncio

from app.services.chat import ChatService, UNKNOWN_ANSWER
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


class FakeChatClient:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if system_prompt == "":
            raise AssertionError("system prompt must not be empty")
        if "Context" not in user_prompt:
            raise AssertionError("context section must be present")
        return "Revenue was 20."


def test_chat_returns_unknown_without_evidence() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClient(),
    )

    result = asyncio.run(service.answer_question("What is revenue?"))

    assert result.grounded is False
    assert result.answer == UNKNOWN_ANSWER
    assert result.citations == []
    assert result.retrieved_count == 0


def test_chat_returns_grounded_answer_with_citations() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatClient(),
    )

    result = asyncio.run(service.answer_question("What is revenue?"))

    assert result.grounded is True
    assert result.answer == "Revenue was 20."
    assert result.retrieved_count == 1
    assert result.citations[0]["filename"] == "a.txt"
