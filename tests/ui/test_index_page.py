import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult
from app.services.ingest import IngestResult


class FakeIngestService:
    async def ingest_upload(self, upload) -> IngestResult:
        return IngestResult(doc_id="doc-123", chunks_indexed=3)


class FakeChatService:
    async def answer_question(self, question: str, history) -> ChatResult:
        return ChatResult(
            answer="ok",
            citations=[],
            grounded=False,
            retrieved_count=0,
        )

    async def stream_answer_question(self, question: str, history):
        yield "ok"


class FakeDocumentService:
    def list_documents(self):
        return [{"doc_id": "doc-123", "filename": "a.txt", "chunks_indexed": 3}]

    def delete_document(self, doc_id: str):
        raise AssertionError("Document service should not be called in index test")


def test_index_page_has_upload_and_chat(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.get("/")
    html = response.text

    assert response.status_code == 200
    assert 'id="upload-form"' in html
    assert 'id="chat-form"' in html
    assert 'id="documents-list"' in html
    assert 'id="refresh-documents"' in html
    assert 'id="chat-history-select"' in html
    assert 'id="clear-chat"' in html
    assert "New Chat" in html
    assert "let conversationHistory = [];" in html
    assert "history" in html
    assert "renderMarkdown" in html
    assert 'fetch("/chat",' in html
    assert 'fetch("/chat/stream")' not in html
    assert '/static/js/chat_highlight.js' in html
    assert "migrateLegacyChatState" in html
    assert "Palette: Red, Gray, Black, White" not in html
    assert 'id="documents-panel"' not in html
    assert 'id="nav-chat"' not in html
    assert 'id="nav-documents"' not in html
    assert "flex justify-end" in html
    assert "flex justify-start" in html


def test_index_page_preserves_markdown_line_breaks(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.get("/")
    html = response.text

    assert response.status_code == 200
    assert '.replace(/[ \\t]{2,}/g, " ")' in html
    assert '.replace(/\\s{2,}/g, " ")' not in html
