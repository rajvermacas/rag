"""Text chunking utilities for indexing."""

import logging


logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk text with deterministic overlap."""
    if text == "":
        raise ValueError("text must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    logger.info(
        "chunk_text_started text_length=%s chunk_size=%s overlap=%s",
        len(text),
        chunk_size,
        overlap,
    )
    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        start += step

    logger.info("chunk_text_completed chunk_count=%s", len(chunks))
    return chunks
