import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.battleground import CompareStreamEvent


class FakeIngestService:
    async def ingest_upload(self, upload):
        raise AssertionError("Ingest service should not be called in battleground API test")


class FakeChatService:
    async def answer_question(self, question: str, history):
        raise AssertionError("Chat service should not be called in battleground API test")

    async def stream_answer_question(self, question: str, history):
        raise AssertionError("Chat stream should not be called in battleground API test")


class FakeDocumentService:
    def list_documents(self):
        return []

    def delete_document(self, doc_id: str):
        raise AssertionError("Document delete should not be called in battleground API test")


class FakeBattlegroundService:
    async def compare_stream(self, question: str, history, model_a: str, model_b: str):
        assert question == "What is revenue?"
        assert model_a == "openai/gpt-4o-mini"
        assert model_b == "anthropic/claude-3.5-sonnet"
        yield CompareStreamEvent(side="A", kind="chunk", chunk="A1", error=None)
        yield CompareStreamEvent(side="B", kind="chunk", chunk="B1", error=None)
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)
        yield CompareStreamEvent(side="B", kind="done", chunk=None, error=None)


class FakeBattlegroundValidationErrorService:
    async def compare_stream(self, question: str, history, model_a: str, model_b: str):
        raise ValueError("model_a and model_b must be different")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


class FakeBattlegroundDisallowedModelService:
    async def compare_stream(self, question: str, history, model_a: str, model_b: str):
        raise ValueError("model_a is not allowed")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


class FakeBattlegroundMustNotBeCalledService:
    async def compare_stream(self, question: str, history, model_a: str, model_b: str):
        raise AssertionError("Battleground service should not be called for invalid payload")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


def _valid_compare_payload() -> dict:
    return {
        "message": "What is revenue?",
        "history": [{"role": "user", "message": "Earlier message"}],
        "model_a": "openai/gpt-4o-mini",
        "model_b": "anthropic/claude-3.5-sonnet",
    }



def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    battleground_service,
) -> TestClient:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
        document_service=FakeDocumentService(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    monkeypatch.setattr(
        main_module,
        "_build_battleground_service",
        lambda services, settings: battleground_service,
    )
    return TestClient(create_app())


def test_get_battleground_models_returns_allowlist(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundService())

    response = client.get("/models/battleground")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
    }


def test_compare_stream_returns_tagged_ndjson_events(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundService())

    response = client.post(
        "/battleground/compare/stream",
        json={
            "message": "What is revenue?",
            "history": [{"role": "user", "message": "Earlier message"}],
            "model_a": "openai/gpt-4o-mini",
            "model_b": "anthropic/claude-3.5-sonnet",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    raw_lines = [line for line in response.text.splitlines() if line.strip() != ""]
    assert len(raw_lines) == 4
    events = [json.loads(line) for line in raw_lines]
    for event in events:
        assert event["side"] in {"A", "B"}
        present_fields = [key for key in ["chunk", "done", "error"] if key in event]
        assert len(present_fields) == 1
    assert events[0] == {"side": "A", "chunk": "A1"}
    assert events[1] == {"side": "B", "chunk": "B1"}
    assert events[2] == {"side": "A", "done": True}
    assert events[3] == {"side": "B", "done": True}


def test_compare_stream_returns_400_for_service_validation_errors(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundValidationErrorService())

    response = client.post(
        "/battleground/compare/stream",
        json={**_valid_compare_payload(), "model_b": "openai/gpt-4o-mini"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model_a and model_b must be different"}


def test_compare_stream_returns_400_for_empty_message(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())

    response = client.post(
        "/battleground/compare/stream",
        json={**_valid_compare_payload(), "message": " "},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid battleground compare payload: message must not be empty"}


@pytest.mark.parametrize(
    ("field_name", "detail_message"),
    [
        ("model_a", "model_a must not be empty"),
        ("model_b", "model_b must not be empty"),
    ],
)
def test_compare_stream_returns_400_for_empty_model_ids(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    detail_message: str,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())

    response = client.post(
        "/battleground/compare/stream",
        json={**_valid_compare_payload(), field_name: "   "},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": f"invalid battleground compare payload: {detail_message}"}


def test_compare_stream_returns_400_for_disallowed_models(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundDisallowedModelService())

    response = client.post(
        "/battleground/compare/stream",
        json={**_valid_compare_payload(), "model_a": "meta/not-allowed"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model_a is not allowed"}


@pytest.mark.parametrize("missing_field", ["message", "history", "model_a", "model_b"])
def test_compare_stream_returns_400_for_missing_required_fields(
    required_env: None, monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())
    payload = _valid_compare_payload()
    del payload[missing_field]

    response = client.post("/battleground/compare/stream", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "detail": f"invalid battleground compare payload: {missing_field} is required"
    }
