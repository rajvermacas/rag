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
        raise AssertionError("Document service should not be called in battleground test")


def _build_index_page_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
        retrieval_service=object(),
        chat_client=object(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    return TestClient(create_app())


def test_battleground_script_streams_multi_turn_battleground_chat_and_preserves_tabs(
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
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["fetchCalls"][0] == {"url": "/models/battleground", "method": "GET", "body": None}
    assert payload["fetchCalls"][1] == {
        "url": "/battleground/compare/stream",
        "method": "POST",
        "body": {
            "message": "Which answer is better?",
            "history_a": [{"role": "user", "message": "Which answer is better?"}],
            "history_b": [{"role": "user", "message": "Which answer is better?"}],
            "model_a": "openai/gpt-4o-mini",
            "model_b": "anthropic/claude-3.5-sonnet",
        },
    }
    assert payload["fetchCalls"][2] == {
        "url": "/battleground/compare/stream",
        "method": "POST",
        "body": {
            "message": "Give me a short follow-up",
            "history_a": [
                {"role": "user", "message": "Which answer is better?"},
                {"role": "assistant", "message": "A says hi"},
                {"role": "user", "message": "Give me a short follow-up"},
            ],
            "history_b": [
                {"role": "user", "message": "Which answer is better?"},
                {"role": "assistant", "message": "B says hi"},
                {"role": "user", "message": "Give me a short follow-up"},
            ],
            "model_a": "openai/gpt-4o-mini",
            "model_b": "anthropic/claude-3.5-sonnet",
        },
    }
    read_snapshots = payload["readSnapshots"]
    assert read_snapshots[0]["status"] == "Comparing models..."
    assert read_snapshots[-1]["status"] == "Comparing models..."
    assert payload["modelSelectStateAfterFirstTurn"] == {"modelADisabled": True, "modelBDisabled": True}
    assert payload["modelAOutput"] == [
        "You: Which answer is better?",
        "Model A: A says hi",
        "You: Give me a short follow-up",
        "Model A: A follow up",
    ]
    assert payload["modelBOutput"] == [
        "You: Which answer is better?",
        "Model B: B says hi",
        "You: Give me a short follow-up",
        "Error: B failed",
    ]
    assert payload["persistedState"]["selectedModelA"] == "openai/gpt-4o-mini"
    assert payload["persistedState"]["selectedModelB"] == "anthropic/claude-3.5-sonnet"
    assert payload["persistedState"]["isModelSelectionLocked"] is True
    assert payload["persistedState"]["historyA"] == [
        {"role": "user", "message": "Which answer is better?"},
        {"role": "assistant", "message": "A says hi"},
        {"role": "user", "message": "Give me a short follow-up"},
        {"role": "assistant", "message": "A follow up"},
    ]
    assert payload["persistedState"]["historyB"] == [
        {"role": "user", "message": "Which answer is better?"},
        {"role": "assistant", "message": "B says hi"},
        {"role": "user", "message": "Give me a short follow-up"},
    ]
    assert payload["finalStatus"] == "Turn complete with side errors on: B."
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
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
    ]
    assert payload["modelBOptions"] == [
        "",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
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

    assert payload["modelAOutput"] == ["You: Which answer is better?", "Model A: A partial"]
    assert payload["modelBOutput"] == ["You: Which answer is better?", "Model B: B partial"]
    assert payload["finalStatus"] == "Battleground stream ended before terminal events for side(s): A, B."
    assert payload["finalStatus"] != "Comparison complete."
