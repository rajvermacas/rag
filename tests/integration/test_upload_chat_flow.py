import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult, NO_DOCUMENT_EVIDENCE
from app.services.ingest import IngestResult


class StatefulFakeIngestService:
    def __init__(self) -> None:
        self.has_upload = False

    async def ingest_upload(self, upload) -> IngestResult:
        self.has_upload = True
        return IngestResult(doc_id="doc-123", chunks_indexed=2)


class StatefulFakeChatService:
    def __init__(self, ingest_service: StatefulFakeIngestService) -> None:
        self._ingest_service = ingest_service

    async def answer_question(self, question: str, history) -> ChatResult:
        if len(history) == 0:
            raise AssertionError("history must be passed to chat service")
        if not self._ingest_service.has_upload:
            return ChatResult(
                answer=(
                    "From uploaded documents:\n"
                    f"{NO_DOCUMENT_EVIDENCE}\n\n"
                    "From general knowledge:\n"
                    "I can still provide a broad answer from general knowledge."
                ),
                citations=[],
                grounded=False,
                retrieved_count=0,
            )
        return ChatResult(
            answer="The document says hello world.",
            citations=[],
            grounded=True,
            retrieved_count=1,
        )

    async def stream_answer_question(self, question: str, history):
        if not self._ingest_service.has_upload:
            yield f"{NO_DOCUMENT_EVIDENCE}"
            return
        yield "The document says "
        yield "hello world."


class StatefulFakeDocumentService:
    def __init__(self, ingest_service: StatefulFakeIngestService) -> None:
        self._ingest_service = ingest_service

    def list_documents(self):
        if not self._ingest_service.has_upload:
            return []
        return [{"doc_id": "doc-123", "filename": "a.txt", "chunks_indexed": 2}]

    def delete_document(self, doc_id: str):
        raise AssertionError("Delete should not be called in flow test")


def test_upload_then_chat_returns_grounded_answer(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_service = StatefulFakeIngestService()
    chat_service = StatefulFakeChatService(ingest_service)
    document_service = StatefulFakeDocumentService(ingest_service)
    fake_services = AppServices(
        ingest_service=ingest_service,
        chat_service=chat_service,
        document_service=document_service,
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    upload_response = client.post(
        "/upload",
        files={"file": ("a.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert upload_response.status_code == 200

    chat_response = client.post(
        "/chat",
        json={
            "message": "What does the document say?",
            "history": [{"role": "user", "message": "Earlier message"}],
        },
    )
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["grounded"] is True
    assert payload["retrieved_count"] == 1
    assert payload["citations"] == []


def test_upload_then_chat_stream_returns_answer(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_service = StatefulFakeIngestService()
    chat_service = StatefulFakeChatService(ingest_service)
    document_service = StatefulFakeDocumentService(ingest_service)
    fake_services = AppServices(
        ingest_service=ingest_service,
        chat_service=chat_service,
        document_service=document_service,
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    upload_response = client.post(
        "/upload",
        files={"file": ("a.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert upload_response.status_code == 200

    stream_response = client.post(
        "/chat/stream",
        json={
            "message": "What does the document say?",
            "history": [{"role": "user", "message": "Earlier message"}],
        },
    )
    assert stream_response.status_code == 200
    assert stream_response.text == "The document says hello world."
