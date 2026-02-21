"""Configuration loading with strict required environment variables."""

from dataclasses import dataclass
import logging
import os

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_int(name: str) -> int:
    raw_value = _require_env(name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for environment variable {name}: {raw_value}"
        ) from exc


def _parse_float(name: str) -> float:
    raw_value = _require_env(name)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float for environment variable {name}: {raw_value}"
        ) from exc


def _parse_required_csv(name: str) -> tuple[str, ...]:
    raw_value = _require_env(name)
    values = [value.strip() for value in raw_value.split(",")]
    if any(value == "" for value in values):
        raise ValueError(f"{name} contains empty model id")
    logger.debug("parsed_required_csv name=%s value_count=%d", name, len(values))
    return tuple(values)


def load_environment_from_dotenv(dotenv_path: str) -> bool:
    if dotenv_path.strip() == "":
        raise ValueError("dotenv_path must not be empty")
    loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
    logger.info("dotenv_load_attempted dotenv_path=%s loaded=%s", dotenv_path, loaded)
    return loaded


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_chat_model: str
    openrouter_embed_model: str
    openrouter_battleground_models: tuple[str, ...]
    chroma_persist_dir: str
    chroma_collection_name: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    min_relevance_score: float
    app_log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=_require_env("OPENROUTER_API_KEY"),
            openrouter_chat_model=_require_env("OPENROUTER_CHAT_MODEL"),
            openrouter_embed_model=_require_env("OPENROUTER_EMBED_MODEL"),
            openrouter_battleground_models=_parse_required_csv(
                "OPENROUTER_BATTLEGROUND_MODELS"
            ),
            chroma_persist_dir=_require_env("CHROMA_PERSIST_DIR"),
            chroma_collection_name=_require_env("CHROMA_COLLECTION_NAME"),
            max_upload_mb=_parse_int("MAX_UPLOAD_MB"),
            chunk_size=_parse_int("CHUNK_SIZE"),
            chunk_overlap=_parse_int("CHUNK_OVERLAP"),
            retrieval_top_k=_parse_int("RETRIEVAL_TOP_K"),
            min_relevance_score=_parse_float("MIN_RELEVANCE_SCORE"),
            app_log_level=_require_env("APP_LOG_LEVEL"),
        )
