import pytest

from app.services.chunking import chunk_text


def test_chunk_text_overlap() -> None:
    text = "a" * 30
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["a" * 10, "a" * 10, "a" * 10, "a" * 6]


def test_chunk_text_empty_raises() -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        chunk_text("", chunk_size=10, overlap=2)


def test_chunk_text_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("abc", chunk_size=3, overlap=3)
