import json
import shutil
import subprocess

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


def _run_remove_citation_artifacts(script_source: str, input_text: str) -> str:
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("node executable is required for ui common.js behavior test")

    harness = f"""
const commonScript = {json.dumps(script_source)};
const inputText = {json.dumps(input_text)};
globalThis.window = globalThis;
globalThis.marked = {{
  setOptions: () => undefined,
  parse: (value) => value,
}};
globalThis.DOMPurify = {{
  sanitize: (value) => value,
}};
eval(commonScript);
const result = window.RagCommon.removeCitationArtifacts(inputText);
if (typeof result !== "string") {{
  throw new Error("removeCitationArtifacts must return a string");
}}
process.stdout.write(JSON.stringify({{ result }}));
"""
    completed = subprocess.run(
        [node_path, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        error_output = completed.stderr.strip()
        raise RuntimeError(f"node harness failed for common.js: {error_output}")

    payload = json.loads(completed.stdout)
    if "result" not in payload:
        raise RuntimeError("node harness output missing 'result'")
    result = payload["result"]
    if not isinstance(result, str):
        raise RuntimeError("node harness output 'result' must be a string")
    return result


def test_index_page_has_chat_and_battleground_scaffolds(
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

    response = client.get("/")
    html = response.text

    assert response.status_code == 200
    assert 'id="upload-form"' in html
    assert 'id="chat-form"' in html
    assert 'id="nav-chat"' in html
    assert 'id="nav-battleground"' in html
    assert 'id="battleground-form"' in html
    assert 'id="model-a-select"' in html
    assert 'id="model-b-select"' in html
    assert 'id="documents-list"' in html
    assert 'id="refresh-documents"' in html
    assert 'id="chat-history-select"' in html
    assert 'id="clear-chat"' in html
    assert "New Chat" in html
    assert '<script src="/static/js/common.js"></script>' in html
    assert '<script src="/static/js/chat.js"></script>' in html
    assert '<script src="/static/js/battleground.js"></script>' in html
    assert "let conversationHistory = [];" not in html
    assert "Palette: Red, Gray, Black, White" not in html
    assert 'id="documents-panel"' not in html
    assert 'id="nav-documents"' not in html


def test_common_script_removes_citation_artifacts_without_collapsing_newlines(
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

    response = client.get("/static/js/common.js")
    assert response.status_code == 200
    cleaned = _run_remove_citation_artifacts(
        response.text,
        "Line one    with   spaces\n[source #1]Line two\t\twith tabs",
    )
    assert cleaned == "Line one with spaces\nLine two with tabs"


def test_chat_script_uses_updated_default_greeting_for_initial_and_new_chat(
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

    response = client.get("/static/js/chat.js")
    assert response.status_code == 200
    script_source = response.text
    assert 'const DEFAULT_CHAT_GREETING = "Hello! How can I assist you today?";' in script_source
    assert script_source.count("appendAssistantMessage(DEFAULT_CHAT_GREETING);") == 2
    assert (
        "Hello. I can answer questions from uploaded files and show indexed files under the upload panel."
        not in script_source
    )
