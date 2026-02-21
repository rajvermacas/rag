"""Retrieval service and relevance gating."""

import logging
from typing import Protocol

from app.services.vector_store import IndexedChunk


logger = logging.getLogger(__name__)


class EmbedClient(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed input text strings."""


class VectorStore(Protocol):
    def query(self, query_embedding: list[float], top_k: int) -> list[IndexedChunk]:
        """Query vector store by embedding."""


def filter_by_relevance(
    results: list[IndexedChunk], min_relevance_score: float
) -> list[IndexedChunk]:
    if len(results) == 0:
        raise ValueError("retrieval returned no results")
    filtered = [result for result in results if result.score >= min_relevance_score]
    if len(filtered) == 0:
        raise ValueError("no results passed relevance threshold")
    return filtered


class RetrievalService:
    """Retrieve context chunks for a user question."""

    def __init__(
        self,
        embed_client: EmbedClient,
        vector_store: VectorStore,
        top_k: int,
        min_relevance_score: float,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self._embed_client = embed_client
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_relevance_score = min_relevance_score

    async def retrieve(self, question: str) -> list[IndexedChunk]:
        if question.strip() == "":
            raise ValueError("question must not be empty")
        logger.info("retrieval_started question_length=%s", len(question))
        query_embedding = (await self._embed_client.embed_texts([question]))[0]
        raw_results = self._vector_store.query(query_embedding, self._top_k)
        filtered_results = filter_by_relevance(raw_results, self._min_relevance_score)
        logger.info("retrieval_completed result_count=%s", len(filtered_results))
        return filtered_results
