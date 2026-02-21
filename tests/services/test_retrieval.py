import asyncio

import pytest

from app.services.retrieval import RetrievalService, filter_by_relevance
from app.services.vector_store import IndexedChunk


def test_filter_by_relevance_drops_weak_results() -> None:
    results = [
        IndexedChunk(
            doc_id="d1",
            filename="a.txt",
            chunk_id="0",
            text="high",
            score=0.91,
            page=None,
        ),
        IndexedChunk(
            doc_id="d1",
            filename="a.txt",
            chunk_id="1",
            text="low",
            score=0.42,
            page=None,
        ),
    ]
    filtered = filter_by_relevance(results, 0.6)
    assert len(filtered) == 1
    assert filtered[0].score == 0.91


def test_filter_by_relevance_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="retrieval returned no results"):
        filter_by_relevance([], 0.6)


def test_filter_by_relevance_no_matches_raises() -> None:
    results = [
        IndexedChunk(
            doc_id="d1",
            filename="a.txt",
            chunk_id="0",
            text="low",
            score=0.12,
            page=None,
        )
    ]
    with pytest.raises(ValueError, match="no results passed relevance threshold"):
        filter_by_relevance(results, 0.6)


class FakeEmbedClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if texts != ["What is in the doc?"]:
            raise AssertionError("unexpected text payload")
        return [[0.1, 0.2, 0.3]]


class FakeVectorStore:
    def query(self, query_embedding: list[float], top_k: int) -> list[IndexedChunk]:
        if query_embedding != [0.1, 0.2, 0.3]:
            raise AssertionError("unexpected embedding")
        if top_k != 3:
            raise AssertionError("unexpected top_k")
        return [
            IndexedChunk(
                doc_id="d1",
                filename="a.txt",
                chunk_id="0",
                text="strong result",
                score=0.9,
                page=None,
            ),
            IndexedChunk(
                doc_id="d1",
                filename="a.txt",
                chunk_id="1",
                text="weak result",
                score=0.2,
                page=None,
            ),
        ]


def test_retrieval_service_returns_filtered_results() -> None:
    retrieval_service = RetrievalService(
        embed_client=FakeEmbedClient(),
        vector_store=FakeVectorStore(),
        top_k=3,
        min_relevance_score=0.5,
    )
    results = asyncio.run(retrieval_service.retrieve("What is in the doc?"))
    assert len(results) == 1
    assert results[0].text == "strong result"
