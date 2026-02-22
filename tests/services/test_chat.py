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


class FakeChatRouterNoEvidence:
    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id != "lab_vllm":
            raise ValueError("backend_id is not allowed")
        return "openai_compatible"

    async def generate_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if backend_id != "lab_vllm":
            raise AssertionError("chat service must pass selected backend_id")
        if model != "openai/gpt-4o-mini":
            raise AssertionError("chat service must pass selected model")
        if NO_DOCUMENT_EVIDENCE not in system_prompt:
            raise AssertionError("missing explicit no-evidence guidance")
        if "Conversation history:\nuser: Earlier message" not in user_prompt:
            raise AssertionError("conversation history should be present in user prompt")
        return f"{NO_DOCUMENT_EVIDENCE} Revenue can refer to total income before expenses."

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        yield await self.generate_chat_response_with_backend(
            backend_id,
            model,
            system_prompt,
            user_prompt,
        )


class FakeChatRouterWithEvidence:
    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id != "lab_vllm":
            raise ValueError("backend_id is not allowed")
        return "openai_compatible"

    async def generate_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if backend_id != "lab_vllm":
            raise AssertionError("chat service must pass selected backend_id")
        if model != "openai/gpt-4o-mini":
            raise AssertionError("chat service must pass selected model")
        if "do not include citations" not in system_prompt:
            raise AssertionError("missing no-citation requirement")
        return "Revenue was 20. [a.txt#0] Revenue is commonly calculated before expenses."

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        if backend_id != "lab_vllm":
            raise AssertionError("chat service must pass selected backend_id")
        if model != "openai/gpt-4o-mini":
            raise AssertionError("chat service must pass selected model")
        yield "Revenue was "
        yield "20. [a.txt#0]"


class FakeChatRouterNotExpected:
    def get_provider_for_backend(self, backend_id: str) -> str:
        if backend_id != "lab_vllm":
            raise ValueError("backend_id is not allowed")
        return "openai_compatible"

    async def generate_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError("chat router should not be called for inventory questions")

    async def stream_chat_response_with_backend(
        self,
        backend_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ):
        raise AssertionError("chat router should not be called for inventory questions")
        yield ""


def test_chat_returns_unknown_without_evidence() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatRouterNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(
        service.answer_question(
            question="What is revenue?",
            history=history,
            backend_id="lab_vllm",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.grounded is False
    assert NO_DOCUMENT_EVIDENCE in result.answer
    assert result.citations == []
    assert result.retrieved_count == 0


def test_chat_returns_grounded_answer_without_citations() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatRouterWithEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(
        service.answer_question(
            question="What is revenue?",
            history=history,
            backend_id="lab_vllm",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.grounded is True
    assert "[a.txt#0]" not in result.answer
    assert result.retrieved_count == 1
    assert result.citations == []


def test_chat_stream_returns_chunks() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalWithEvidence(),
        chat_client=FakeChatRouterWithEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    async def collect_chunks() -> list[str]:
        stream = await service.stream_answer_question(
            question="What is revenue?",
            history=history,
            backend_id="lab_vllm",
            model="openai/gpt-4o-mini",
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect_chunks())
    assert len(chunks) == 2


def test_chat_rejects_empty_history_message() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatRouterNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="assistant", message=" ")]

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=history,
                backend_id="lab_vllm",
                model="openai/gpt-4o-mini",
            )
        )
        raise AssertionError("expected ValueError for empty history message")
    except ValueError as exc:
        assert str(exc) == "history message must not be empty"


def test_chat_answers_document_inventory_without_model_call() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatRouterNotExpected(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    result = asyncio.run(
        service.answer_question(
            question="What documents do you have access to?",
            history=history,
            backend_id="lab_vllm",
            model="openai/gpt-4o-mini",
        )
    )

    assert result.grounded is True
    assert result.retrieved_count == 0
    assert "a.txt" in result.answer
    assert "b.pdf" in result.answer


def test_chat_rejects_empty_backend_id() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatRouterNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=history,
                backend_id="   ",
                model="openai/gpt-4o-mini",
            )
        )
        raise AssertionError("expected ValueError for empty backend_id")
    except ValueError as exc:
        assert str(exc) == "backend_id must not be empty"


def test_chat_rejects_empty_model_id() -> None:
    service = ChatService(
        retrieval_service=FakeRetrievalNoEvidence(),
        chat_client=FakeChatRouterNoEvidence(),
        document_service=FakeDocumentService(),
    )
    history = [ConversationTurn(role="user", message="Earlier message")]

    try:
        asyncio.run(
            service.answer_question(
                question="What is revenue?",
                history=history,
                backend_id="lab_vllm",
                model="  ",
            )
        )
        raise AssertionError("expected ValueError for empty model")
    except ValueError as exc:
        assert str(exc) == "model must not be empty"
