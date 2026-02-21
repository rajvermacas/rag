import pytest

from app.services.vector_store import (
    _convert_document_result,
    _convert_query_result,
    _distance_to_relevance_score,
)


def test_distance_to_relevance_score_requires_non_negative_distance() -> None:
    with pytest.raises(ValueError, match="distance must be non-negative"):
        _distance_to_relevance_score(-0.1)


def test_distance_to_relevance_score_is_reciprocal() -> None:
    assert _distance_to_relevance_score(0.0) == 1.0
    assert _distance_to_relevance_score(1.0) == 0.5
    assert _distance_to_relevance_score(3.0) == 0.25


def test_convert_query_result_maps_distance_to_relevance_score() -> None:
    raw_result = {
        "documents": [["chunk text"]],
        "metadatas": [[{"doc_id": "doc-1", "filename": "a.txt", "chunk_id": 0}]],
        "distances": [[1.25]],
    }

    chunks = _convert_query_result(raw_result)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].filename == "a.txt"
    assert chunks[0].chunk_id == "0"
    assert chunks[0].text == "chunk text"
    assert chunks[0].score == pytest.approx(1.0 / 2.25)


def test_convert_document_result_groups_by_doc_id() -> None:
    raw_result = {
        "metadatas": [
            {"doc_id": "doc-2", "filename": "b.pdf", "chunk_id": 0},
            {"doc_id": "doc-1", "filename": "a.txt", "chunk_id": 0},
            {"doc_id": "doc-1", "filename": "a.txt", "chunk_id": 1},
        ]
    }

    documents = _convert_document_result(raw_result)

    assert len(documents) == 2
    assert documents[0].doc_id == "doc-1"
    assert documents[0].filename == "a.txt"
    assert documents[0].chunks_indexed == 2
    assert documents[1].doc_id == "doc-2"
    assert documents[1].chunks_indexed == 1
