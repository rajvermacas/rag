"""Vector store wrapper around ChromaDB."""

from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedChunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str
    score: float
    page: int | None


class ChromaVectorStore:
    """ChromaDB persistence and retrieval service."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        if persist_dir.strip() == "":
            raise ValueError("persist_dir must not be empty")
        if collection_name.strip() == "":
            raise ValueError("collection_name must not be empty")

        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = self._load_collection()

    def upsert_chunks(
        self, doc_id: str, filename: str, chunks: list[str], embeddings: list[list[float]]
    ) -> int:
        if doc_id.strip() == "":
            raise ValueError("doc_id must not be empty")
        if filename.strip() == "":
            raise ValueError("filename must not be empty")
        if len(chunks) == 0:
            raise ValueError("chunks must not be empty")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        logger.info("chroma_upsert_started doc_id=%s chunk_count=%s", doc_id, len(chunks))
        ids = [f"{doc_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "chunk_id": index}
            for index in range(len(chunks))
        ]
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("chroma_upsert_completed doc_id=%s chunk_count=%s", doc_id, len(chunks))
        return len(chunks)

    def query(self, query_embedding: list[float], top_k: int) -> list[IndexedChunk]:
        if len(query_embedding) == 0:
            raise ValueError("query_embedding must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        logger.info("chroma_query_started top_k=%s", top_k)
        raw_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = _convert_query_result(raw_result)
        logger.info("chroma_query_completed result_count=%s", len(chunks))
        return chunks

    def _load_collection(self):
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency for vector store: chromadb") from exc

        client = chromadb.PersistentClient(path=self._persist_dir)
        return client.get_or_create_collection(name=self._collection_name)


def _convert_query_result(raw_result: dict) -> list[IndexedChunk]:
    documents = raw_result["documents"][0]
    metadatas = raw_result["metadatas"][0]
    distances = raw_result["distances"][0]
    items: list[IndexedChunk] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index]
        distance = distances[index]
        items.append(
            IndexedChunk(
                doc_id=str(metadata["doc_id"]),
                filename=str(metadata["filename"]),
                chunk_id=str(metadata["chunk_id"]),
                text=str(document),
                score=_distance_to_relevance_score(float(distance)),
                page=metadata.get("page"),
            )
        )
    return items


def _distance_to_relevance_score(distance: float) -> float:
    if distance < 0.0:
        raise ValueError(f"distance must be non-negative, got: {distance}")
    return 1.0 / (1.0 + distance)
