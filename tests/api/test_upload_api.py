import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.ingest import IngestResult


class FakeIngestService:
    async def ingest_upload(self, upload) -> IngestResult:
        if upload.filename == "bad.csv":
            raise ValueError("Unsupported file type: text/csv")
        return IngestResult(doc_id="doc-123", chunks_indexed=3)


class FakeChatService:
    async def answer_question(self, question: str, history):
        raise AssertionError("Chat service should not be called in upload test")


class FakeDocumentService:
    def list_documents(self):
        return []

    def delete_document(self, doc_id: str):
        raise AssertionError("Document service should not be called in upload test")


def test_upload_txt_indexes_document(required_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    payload = io.BytesIO(b"rag text")
    response = client.post("/upload", files={"file": ("a.txt", payload, "text/plain")})

    assert response.status_code == 200
    assert response.json()["doc_id"] == "doc-123"
    assert response.json()["chunks_indexed"] == 3


def test_upload_returns_400_for_validation_error(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    payload = io.BytesIO(b"x,y")
    response = client.post("/upload", files={"file": ("bad.csv", payload, "text/csv")})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
