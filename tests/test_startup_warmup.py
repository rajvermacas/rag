from types import MappingProxyType, SimpleNamespace

import pytest

from app.config import ChatBackendProfile
from app.startup_warmup import build_backend_model_pairs, warm_up_runtime_dependencies
import app.startup_warmup as startup_warmup_module


class FakeWarmupQueryService:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def warm_up(self, backend_model_pairs: tuple[tuple[str, str], ...]) -> None:
        self.calls.append(backend_model_pairs)


def _build_profiles() -> MappingProxyType[str, ChatBackendProfile]:
    return MappingProxyType(
        {
            "lab_vllm": ChatBackendProfile(
                backend_id="lab_vllm",
                provider="openrouter",
                models=("openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"),
                api_key="test-openrouter-chat-key",
                base_url=None,
                azure_endpoint=None,
                azure_api_version=None,
                azure_deployments=MappingProxyType({}),
            ),
            "azure_prod": ChatBackendProfile(
                backend_id="azure_prod",
                provider="azure_openai",
                models=("gpt-4o-mini",),
                api_key="test-azure-key",
                base_url=None,
                azure_endpoint="https://azure-openai.example.com",
                azure_api_version="2024-10-21",
                azure_deployments=MappingProxyType({"gpt-4o-mini": "chat-gpt4o-mini"}),
            ),
        }
    )


def test_build_backend_model_pairs_collects_all_models() -> None:
    settings_like = SimpleNamespace(chat_backend_profiles=_build_profiles())

    pairs = build_backend_model_pairs(settings_like.chat_backend_profiles)

    assert pairs == (
        ("lab_vllm", "openai/gpt-4o-mini"),
        ("lab_vllm", "anthropic/claude-3.5-sonnet"),
        ("azure_prod", "gpt-4o-mini"),
    )


def test_build_backend_model_pairs_rejects_empty_profile_set() -> None:
    settings_like = SimpleNamespace(chat_backend_profiles=MappingProxyType({}))

    with pytest.raises(
        ValueError,
        match="chat backend profiles must include at least one backend/model",
    ):
        build_backend_model_pairs(settings_like.chat_backend_profiles)


def test_warm_up_runtime_dependencies_imports_transformers_and_primes_query_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_query_service = FakeWarmupQueryService()
    import_calls: list[str] = []

    monkeypatch.setattr(
        startup_warmup_module,
        "_import_transformers_if_available",
        lambda: import_calls.append("transformers_imported"),
    )

    backend_model_pairs = (("lab_vllm", "openai/gpt-4o-mini"),)
    warm_up_runtime_dependencies(fake_query_service, backend_model_pairs)

    assert import_calls == ["transformers_imported"]
    assert fake_query_service.calls == [backend_model_pairs]
