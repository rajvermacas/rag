import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.documents import DocumentSummary


class FakeIngestService:
    async def ingest_upload(self, upload):
        raise AssertionError("Ingest service should not be called in documents test")


class FakeChatService:
    async def answer_question(self, question: str, history, backend_id: str, model: str):
        raise AssertionError("Chat service should not be called in documents test")

    async def stream_answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ):
        raise AssertionError("Chat stream should not be called in documents test")


class FakeDocumentService:
    def __init__(self) -> None:
        self._docs = [
            DocumentSummary(doc_id="doc-1", filename="policy.txt", chunks_indexed=2),
            DocumentSummary(doc_id="doc-2", filename="pricing.pdf", chunks_indexed=4),
        ]

    def list_documents(self) -> list[DocumentSummary]:
        return list(self._docs)

    def delete_document(self, doc_id: str) -> int:
        for index, document in enumerate(self._docs):
            if document.doc_id == doc_id:
                removed = self._docs.pop(index)
                return int(removed.chunks_indexed)
        raise ValueError(f"document not found: {doc_id}")


def test_list_documents_returns_all_uploaded_documents(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_provider_router=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.get("/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"][0]["filename"] == "policy.txt"
    assert payload["documents"][1]["doc_id"] == "doc-2"


def test_delete_document_removes_indexed_document(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_document_service = FakeDocumentService()
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=fake_document_service,
        retrieval_service=object(),
        chat_provider_router=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    delete_response = client.delete("/documents/doc-1")

    assert delete_response.status_code == 200
    assert delete_response.json()["doc_id"] == "doc-1"
    assert delete_response.json()["chunks_deleted"] == 2

    list_response = client.get("/documents")
    payload = list_response.json()
    assert len(payload["documents"]) == 1
    assert payload["documents"][0]["doc_id"] == "doc-2"


def test_delete_document_returns_404_when_doc_missing(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_provider_router=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.delete("/documents/missing-doc")

    assert response.status_code == 404
    assert "document not found" in response.json()["detail"]
