"""Configuration loading with strict required environment variables."""

from dataclasses import dataclass
import os


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


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_chat_model: str
    openrouter_embed_model: str
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
            chroma_persist_dir=_require_env("CHROMA_PERSIST_DIR"),
            chroma_collection_name=_require_env("CHROMA_COLLECTION_NAME"),
            max_upload_mb=_parse_int("MAX_UPLOAD_MB"),
            chunk_size=_parse_int("CHUNK_SIZE"),
            chunk_overlap=_parse_int("CHUNK_OVERLAP"),
            retrieval_top_k=_parse_int("RETRIEVAL_TOP_K"),
            min_relevance_score=_parse_float("MIN_RELEVANCE_SCORE"),
            app_log_level=_require_env("APP_LOG_LEVEL"),
        )
