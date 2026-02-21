import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import AppServices, create_app
from app.services.chat import ChatResult, UNKNOWN_ANSWER


class FakeIngestService:
    async def ingest_upload(self, upload):
        raise AssertionError("Ingest service should not be called in chat test")


class FakeChatService:
    async def answer_question(self, question: str) -> ChatResult:
        if question == "unknown":
            return ChatResult(
                answer=UNKNOWN_ANSWER,
                citations=[],
                grounded=False,
                retrieved_count=0,
            )
        return ChatResult(
            answer="Revenue is 20.",
            citations=[
                {
                    "doc_id": "doc-1",
                    "filename": "a.txt",
                    "chunk_id": "0",
                    "score": 0.92,
                    "page": None,
                }
            ],
            grounded=True,
            retrieved_count=1,
        )


def test_chat_returns_unknown_when_no_evidence(
    required_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.post("/chat", json={"message": "unknown"})

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["answer"] == UNKNOWN_ANSWER
    assert response.json()["citations"] == []


def test_chat_returns_grounded_answer(required_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_services = AppServices(
        ingest_service=FakeIngestService(),
        chat_service=FakeChatService(),
    )
    monkeypatch.setattr(main_module, "_build_services", lambda settings: fake_services)
    client = TestClient(create_app())

    response = client.post("/chat", json={"message": "What is revenue?"})

    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["retrieved_count"] == 1
    assert response.json()["citations"][0]["filename"] == "a.txt"
