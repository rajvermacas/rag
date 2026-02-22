import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.battleground import CompareStreamEvent
from app.services.chat_provider_models import ChatModelOption


class FakeIngestService:
    async def ingest_upload(self, upload):
        raise AssertionError("Ingest service should not be called in battleground API test")


class FakeChatService:
    async def answer_question(self, question: str, history, backend_id: str, model: str):
        raise AssertionError("Chat service should not be called in battleground API test")

    async def stream_answer_question(
        self,
        question: str,
        history,
        backend_id: str,
        model: str,
    ):
        raise AssertionError("Chat stream should not be called in battleground API test")


class FakeDocumentService:
    def list_documents(self):
        return []

    def delete_document(self, doc_id: str):
        raise AssertionError("Document delete should not be called in battleground API test")


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
            ChatModelOption(
                backend_id="lab_vllm",
                provider="openai_compatible",
                model="anthropic/claude-3.5-sonnet",
                label="lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet",
            ),
        )


class FakeBattlegroundService:
    async def compare_stream(
        self,
        question: str,
        history,
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ):
        assert question == "What is revenue?"
        assert model_a_backend_id == "lab_vllm"
        assert model_a == "openai/gpt-4o-mini"
        assert model_b_backend_id == "lab_vllm"
        assert model_b == "anthropic/claude-3.5-sonnet"
        yield CompareStreamEvent(side="A", kind="chunk", chunk="A1", error=None)
        yield CompareStreamEvent(side="B", kind="chunk", chunk="B1", error=None)
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)
        yield CompareStreamEvent(side="B", kind="done", chunk=None, error=None)


class FakeBattlegroundValidationErrorService:
    async def compare_stream(
        self,
        question: str,
        history,
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ):
        raise ValueError("model_a and model_b must be different")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


class FakeBattlegroundDisallowedModelService:
    async def compare_stream(
        self,
        question: str,
        history,
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ):
        raise ValueError("model_a is not allowed for backend_id")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


class FakeBattlegroundMustNotBeCalledService:
    async def compare_stream(
        self,
        question: str,
        history,
        model_a_backend_id: str,
        model_a: str,
        model_b_backend_id: str,
        model_b: str,
    ):
        raise AssertionError("Battleground service should not be called for invalid payload")
        yield CompareStreamEvent(side="A", kind="done", chunk=None, error=None)


def _valid_compare_payload() -> dict:
    return {
        "message": "What is revenue?",
        "history": [{"role": "user", "message": "Earlier message"}],
        "model_a_backend_id": "lab_vllm",
        "model_a": "openai/gpt-4o-mini",
        "model_b_backend_id": "lab_vllm",
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
        retrieval_service=object(),
        chat_provider_router=FakeChatProviderRouter(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    monkeypatch.setattr(
        main_module,
        "_build_battleground_service",
        lambda services: battleground_service,
    )
    return TestClient(create_app())


def test_get_battleground_models_returns_provider_aware_options(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundService())

    response = client.get("/models/battleground")

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
            {
                "backend_id": "lab_vllm",
                "provider": "openai_compatible",
                "model": "anthropic/claude-3.5-sonnet",
                "label": "lab_vllm (openai_compatible) · anthropic/claude-3.5-sonnet",
            },
        ]
    }


def test_compare_stream_returns_tagged_ndjson_events(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundService())

    response = client.post(
        "/battleground/compare/stream",
        json=_valid_compare_payload(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    raw_lines = [line for line in response.text.splitlines() if line.strip() != ""]
    assert len(raw_lines) == 4
    events = [json.loads(line) for line in raw_lines]
    assert events[0] == {"side": "A", "chunk": "A1"}
    assert events[1] == {"side": "B", "chunk": "B1"}
    assert events[2] == {"side": "A", "done": True}
    assert events[3] == {"side": "B", "done": True}


def test_compare_stream_returns_400_for_same_backend_model_pair(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundValidationErrorService())

    response = client.post(
        "/battleground/compare/stream",
        json={
            **_valid_compare_payload(),
            "model_b_backend_id": "lab_vllm",
            "model_b": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model_a and model_b must be different"}


def test_compare_stream_returns_400_for_disallowed_models(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundDisallowedModelService())

    response = client.post(
        "/battleground/compare/stream",
        json={**_valid_compare_payload(), "model_a": "meta/not-allowed"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model is not allowed for backend_id"}


@pytest.mark.parametrize(
    "missing_field",
    [
        "message",
        "history",
        "model_a_backend_id",
        "model_a",
        "model_b_backend_id",
        "model_b",
    ],
)
def test_compare_stream_returns_400_for_missing_required_fields(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())
    payload = _valid_compare_payload()
    del payload[missing_field]

    response = client.post("/battleground/compare/stream", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "detail": f"invalid battleground compare payload: {missing_field} is required"
    }


def test_compare_stream_returns_400_for_non_object_json_payloads(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())

    response = client.post("/battleground/compare/stream", json=[])

    assert response.status_code == 400
    assert response.json() == {
        "detail": "invalid battleground compare payload: payload must be a JSON object"
    }


def test_compare_stream_returns_clear_400_for_malformed_json(
    required_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _build_client(monkeypatch, FakeBattlegroundMustNotBeCalledService())

    response = client.post(
        "/battleground/compare/stream",
        content='{"message":"bad-json"',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "invalid battleground compare payload: payload must be valid JSON"
    }


def test_serialize_battleground_event_rejects_non_string_side() -> None:
    with pytest.raises(ValueError, match="battleground event side must be a string"):
        main_module._serialize_battleground_event(
            CompareStreamEvent(side=1, kind="done", chunk=None, error=None)  # type: ignore[arg-type]
        )
