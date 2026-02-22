import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult
from app.services.ingest import IngestResult
from tests.ui._battleground_harnesses import (
    _BATTLEGROUND_BOOTSTRAP_FAILURE_HARNESS_TEMPLATE,
    _BATTLEGROUND_STREAM_HARNESS_TEMPLATE,
    _BATTLEGROUND_TRUNCATED_STREAM_HARNESS_TEMPLATE,
    _BATTLEGROUND_VALIDATION_HARNESS_TEMPLATE,
    _run_battleground_harness,
)


class FakeIngestService:
    async def ingest_upload(self, upload) -> IngestResult:
        return IngestResult(doc_id="doc-123", chunks_indexed=3)


class FakeChatService:
    async def answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ) -> ChatResult:
        return ChatResult(
            answer="ok",
            citations=[],
            grounded=False,
            retrieved_count=0,
        )

    async def stream_answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ):
        yield "ok"


class FakeDocumentService:
    def list_documents(self):
        return [{"doc_id": "doc-123", "filename": "a.txt", "chunks_indexed": 3}]

    def delete_document(self, doc_id: str):
        raise AssertionError("Document service should not be called in battleground test")


def _build_index_page_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_provider_router=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    return TestClient(create_app())


def test_battleground_script_loads_models_renders_markdown_supports_follow_ups_and_preserves_tabs(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_STREAM_HARNESS_TEMPLATE,
        "ui battleground.js stream behavior test",
    )

    assert payload["modelAOptions"] == [
        "",
        "lab_vllm||openai/gpt-4o-mini",
        "lab_vllm||anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "lab_vllm||openai/gpt-4o-mini",
        "lab_vllm||anthropic/claude-3.5-sonnet",
    ]
    assert payload["fetchCalls"][0] == {"url": "/models/battleground", "method": "GET", "body": None}
    assert payload["fetchCalls"][1] == {
        "url": "/battleground/compare/stream",
        "method": "POST",
        "body": {
            "message": "Which answer is better?",
            "history": [],
            "model_a_backend_id": "lab_vllm",
            "model_a": "openai/gpt-4o-mini",
            "model_b_backend_id": "lab_vllm",
            "model_b": "anthropic/claude-3.5-sonnet",
        },
    }
    assert payload["fetchCalls"][2]["url"] == "/battleground/compare/stream"
    assert payload["firstRequestBody"] == payload["fetchCalls"][1]["body"]
    assert payload["secondRequestBody"] == payload["fetchCalls"][2]["body"]
    assert payload["secondRequestBody"]["message"] == "Can you follow up with examples?"
    assert len(payload["secondRequestBody"]["history"]) > 0
    assert payload["secondRequestBody"]["history"][0]["role"] == "user"
    assert payload["secondRequestBody"]["history"][0]["message"] == "Which answer is better?"
    assert payload["modelATitle"] == "Model A · lab_vllm (openai_compatible) · openai/gpt-4o-mini"
    assert (
        payload["modelBTitle"]
        == "Model B · lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet"
    )
    read_snapshots = payload["readSnapshots"]
    assert read_snapshots[0]["inputValue"] == ""
    assert "Thinking..." in read_snapshots[0]["modelAHtml"]
    assert "Thinking..." in read_snapshots[0]["modelBHtml"]
    assert "Which answer is better?" in payload["modelAHtml"]
    assert "Which answer is better?" in payload["modelBHtml"]
    assert "Can you follow up with examples?" in payload["modelAHtml"]
    assert "Can you follow up with examples?" in payload["modelBHtml"]
    assert "max-w-[85%] rounded-xl border border-zinc-300 bg-zinc-200/90 p-3" in payload["modelAHtml"]
    assert "max-w-[85%] rounded-xl border border-zinc-300 bg-zinc-200/90 p-3" in payload["modelBHtml"]
    assert "w-full max-w-full rounded-xl border border-red-200 bg-white p-3 shadow-sm" in payload["modelAHtml"]
    assert "w-full max-w-full rounded-xl border border-red-200 bg-white p-3 shadow-sm" in payload["modelBHtml"]
    assert payload["modelAHtml"].count("<strong>hi</strong>") == 2
    assert payload["modelBHtml"].count("<em>hi</em>") == 2
    assert "Done." not in payload["modelAHtml"]
    assert "Done." not in payload["modelBHtml"]
    assert payload["modelBHtml"].count("Error: B failed") == 2
    assert payload["finalStatus"] == "Comparison complete with side errors on: B."
    assert payload["afterBattlegroundTab"] == {
        "chatHidden": True,
        "battlegroundHidden": False,
        "chatSelected": "false",
        "battlegroundSelected": "true",
    }
    assert payload["afterChatTab"] == {
        "chatHidden": False,
        "battlegroundHidden": True,
        "chatSelected": "true",
        "battlegroundSelected": "false",
    }


def test_battleground_script_fails_fast_on_invalid_client_inputs(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_VALIDATION_HARNESS_TEMPLATE,
        "ui battleground.js validation behavior test",
    )

    assert payload["modelAOptions"] == [
        "",
        "lab_vllm||openai/gpt-4o-mini",
        "lab_vllm||anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "lab_vllm||openai/gpt-4o-mini",
        "lab_vllm||anthropic/claude-3.5-sonnet",
    ]
    assert payload["statuses"] == [
        "Enter a prompt before starting comparison.",
        "Choose a model for Model A.",
        "Choose a model for Model B.",
        "Model A and Model B must be different.",
    ]
    assert payload["postCallCount"] == 0


def test_battleground_script_handles_model_loading_failure_without_unhandled_rejection(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_BOOTSTRAP_FAILURE_HARNESS_TEMPLATE,
        "ui battleground.js bootstrap failure test",
    )

    assert payload["status"] == "Battleground model list failed."
    assert payload["unhandledRejection"] is None
    assert payload["errorLogs"]
    assert "battleground_initialization_failed" in payload["errorLogs"][0]


def test_battleground_script_reports_error_when_stream_ends_without_terminal_events(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_index_page_client(monkeypatch)

    common_response = client.get("/static/js/common.js")
    battleground_response = client.get("/static/js/battleground.js")
    assert common_response.status_code == 200
    assert battleground_response.status_code == 200

    payload = _run_battleground_harness(
        common_response.text,
        battleground_response.text,
        _BATTLEGROUND_TRUNCATED_STREAM_HARNESS_TEMPLATE,
        "ui battleground.js truncated stream test",
    )

    assert "Which answer is better?" in payload["modelAHtml"]
    assert "Which answer is better?" in payload["modelBHtml"]
    assert "A partial" in payload["modelAHtml"]
    assert "B partial" in payload["modelBHtml"]
    assert payload["finalStatus"] == "Battleground stream ended before terminal events for side(s): A, B."
    assert payload["finalStatus"] != "Comparison complete."
