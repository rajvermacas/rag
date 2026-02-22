"""Startup warm-up helpers for pre-initializing runtime dependencies."""

import logging
from typing import Mapping, Protocol

from app.config import ChatBackendProfile


logger = logging.getLogger(__name__)


class QueryServiceWarmup(Protocol):
    def warm_up(self, backend_models: tuple[tuple[str, str], ...]) -> None:
        """Pre-build backend/model runtime state."""


def build_backend_model_pairs(
    chat_backend_profiles: Mapping[str, ChatBackendProfile],
) -> tuple[tuple[str, str], ...]:
    backend_model_pairs: list[tuple[str, str]] = []
    for backend_id, profile in chat_backend_profiles.items():
        for model in profile.models:
            backend_model_pairs.append((backend_id, model))
    if len(backend_model_pairs) == 0:
        raise ValueError("chat backend profiles must include at least one backend/model")
    return tuple(backend_model_pairs)


def warm_up_runtime_dependencies(
    query_service: QueryServiceWarmup,
    backend_model_pairs: tuple[tuple[str, str], ...],
) -> None:
    logger.info(
        "application_runtime_warmup_started backend_model_count=%s",
        len(backend_model_pairs),
    )
    _import_transformers_if_available()
    query_service.warm_up(backend_model_pairs)
    logger.info(
        "application_runtime_warmup_completed backend_model_count=%s",
        len(backend_model_pairs),
    )


def _import_transformers_if_available() -> None:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        logger.info("application_runtime_warmup_transformers_missing")
        return
    logger.info("application_runtime_warmup_transformers_imported")
