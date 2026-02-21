import asyncio

from app.services.chat import ChatService, NO_DOCUMENT_EVIDENCE
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
        if "Context:\n" not in user_prompt:
            raise AssertionError("context section must be present")
        return (
            "From uploaded documents (with citations):\n"
            "No relevant evidence found in uploaded documents.\n\n"
            "From general knowledge (not from uploaded documents):\n"
            "Revenue can refer to total income before expenses."
        )


class FakeChatClientWithEvidence:
    async def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if "[filename#chunk_id]" not in system_prompt:
            raise AssertionError("missing citation format requirement")
        if "a.txt" not in user_prompt:
            raise AssertionError("expected chunk metadata in context")
        return (
            "From uploaded documents (with citations):\n"
            "Revenue was 20. [a.txt#0]\n\n"
            "From general knowledge (not from uploaded documents):\n"
            "Revenue is commonly calculated before subtracting expenses."
        )


def test_chat_returns_unknown_without_evidence() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatClientNoEvidence(),
    )

    result = asyncio.run(service.answer_question("What is revenue?"))

    assert result.grounded is False
    assert NO_DOCUMENT_EVIDENCE in result.answer
    assert result.citations == []
    assert result.retrieved_count == 0


def test_chat_returns_grounded_answer_with_citations() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatClientWithEvidence(),
    )

    result = asyncio.run(service.answer_question("What is revenue?"))

    assert result.grounded is True
    assert "[a.txt#0]" in result.answer
    assert result.retrieved_count == 1
    assert result.citations[0]["filename"] == "a.txt"
