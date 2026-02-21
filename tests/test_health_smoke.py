import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint_exists(required_env: None) -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
