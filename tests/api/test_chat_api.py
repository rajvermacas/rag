import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult, NO_DOCUMENT_EVIDENCE
from app.services.chat_provider_models import ChatModelOption


class FakeIngestService:
    async def ingest_upload(self, upload):
        raise AssertionError("Ingest service should not be called in chat test")


class FakeChatService:
    async def answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ) -> ChatResult:
        if len(history) == 0:
            raise AssertionError("history must be passed to chat service")
        if backend_id != "lab_vllm":
            raise AssertionError("backend_id must be passed to chat service")
        if model != "openai/gpt-4o-mini":
            raise AssertionError("model must be passed to chat service")
        if question == "unknown":
            return ChatResult(
                answer=(
                    f"{NO_DOCUMENT_EVIDENCE} "
                    "I can still provide a high-level answer from general knowledge."
                ),
                citations=[],
                grounded=False,
                retrieved_count=0,
            )
        return ChatResult(
            answer="Revenue is 20.",
            citations=[],
            grounded=True,
            retrieved_count=1,
        )

    async def stream_answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ):
        if len(history) == 0:
            raise AssertionError("history must be passed to chat stream service")
        if backend_id != "lab_vllm":
            raise AssertionError("backend_id must be passed to chat stream service")
        if model != "openai/gpt-4o-mini":
            raise AssertionError("model must be passed to chat stream service")
        if question == "What is revenue?":
            yield "Revenue "
            yield "is 20."
            return
        yield "Unknown question."


class FakeDocumentService:
    def list_documents(self):
        return [{"doc_id": "doc-1", "filename": "a.txt", "chunks_indexed": 1}]

    def delete_document(self, doc_id: str):
        raise AssertionError("Document service should not be called in chat test")


class FakeChatProviderRouter:
    def list_model_options(self) -> tuple[ChatModelOption, ...]:
        return (
            ChatModelOption(
                backend_id="lab_vllm",
                provider="openai_compatible",
                model="openai/gpt-4o-mini",
                label="lab_vllm (openai_compatible) · openai/gpt-4o-mini",
            ),
            ChatModelOption(
                backend_id="azure_prod",
                provider="azure_openai",
                model="gpt-4o-mini",
                label="azure_prod (azure_openai) · gpt-4o-mini",
            ),
        )


def _build_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_provider_router=FakeChatProviderRouter(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    return TestClient(create_app())


def test_chat_returns_unknown_when_no_evidence(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "unknown",
            "history": [{"role": "user", "message": "Earlier message"}],
            "backend_id": "lab_vllm",
            "model": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert NO_DOCUMENT_EVIDENCE in response.json()["answer"]
    assert response.json()["citations"] == []


def test_chat_returns_grounded_answer(required_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "backend_id": "lab_vllm",
            "model": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["retrieved_count"] == 1
    assert response.json()["answer"] == "Revenue is 20."
    assert response.json()["citations"] == []


def test_chat_stream_returns_chunked_answer(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat/stream",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "backend_id": "lab_vllm",
            "model": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    assert response.text == "Revenue is 20."


def test_chat_requires_backend_id_field(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "model": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 422


def test_chat_requires_model_field(required_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "backend_id": "lab_vllm",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_disallowed_backend_model_pair(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "backend_id": "lab_vllm",
            "model": "meta/not-allowed",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model is not allowed for backend_id"}


def test_get_chat_models_returns_provider_aware_options(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch)

    response = client.get("/models/chat")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "backend_id": "lab_vllm",
                "provider": "openai_compatible",
                "model": "openai/gpt-4o-mini",
                "label": "lab_vllm (openai_compatible) · openai/gpt-4o-mini",
            },
            {
                "backend_id": "azure_prod",
                "provider": "azure_openai",
                "model": "gpt-4o-mini",
                "label": "azure_prod (azure_openai) · gpt-4o-mini",
            },
        ]
    }
