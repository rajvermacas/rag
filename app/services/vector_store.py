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


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    filename: str
    chunks_indexed: int


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

    def list_documents(self) -> list[IndexedDocument]:
        logger.info("chroma_list_documents_started")
        raw_result = self._collection.get(include=["metadatas"])
        documents = _convert_document_result(raw_result)
        logger.info("chroma_list_documents_completed document_count=%s", len(documents))
        return documents

    def delete_document(self, doc_id: str) -> int:
        if doc_id.strip() == "":
            raise ValueError("doc_id must not be empty")
        logger.info("chroma_delete_document_started doc_id=%s", doc_id)
        raw_result = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = _extract_ids(raw_result)
        if len(ids) == 0:
            raise ValueError(f"document not found: {doc_id}")
        self._collection.delete(ids=ids)
        logger.info(
            "chroma_delete_document_completed doc_id=%s chunk_count=%s",
            doc_id,
            len(ids),
        )
        return len(ids)

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


def _convert_document_result(raw_result: dict) -> list[IndexedDocument]:
    metadatas = _extract_metadatas(raw_result)
    summaries: dict[str, IndexedDocument] = {}
    for metadata in metadatas:
        doc_id = str(metadata["doc_id"])
        filename = str(metadata["filename"])
        existing_summary = summaries.get(doc_id)
        if existing_summary is None:
            summaries[doc_id] = IndexedDocument(
                doc_id=doc_id,
                filename=filename,
                chunks_indexed=1,
            )
            continue
        if existing_summary.filename != filename:
            raise ValueError(
                f"inconsistent filename for doc_id={doc_id}: "
                f"{existing_summary.filename} != {filename}"
            )
        summaries[doc_id] = IndexedDocument(
            doc_id=existing_summary.doc_id,
            filename=existing_summary.filename,
            chunks_indexed=existing_summary.chunks_indexed + 1,
        )
    return sorted(
        summaries.values(),
        key=lambda document: (document.filename.lower(), document.doc_id),
    )


def _extract_metadatas(raw_result: dict) -> list[dict]:
    if "metadatas" not in raw_result:
        raise ValueError("missing key in Chroma result: metadatas")
    metadatas = raw_result["metadatas"]
    if metadatas is None:
        raise ValueError("metadatas must not be None")
    return [metadata for metadata in metadatas if metadata is not None]


def _extract_ids(raw_result: dict) -> list[str]:
    if "ids" not in raw_result:
        raise ValueError("missing key in Chroma result: ids")
    ids = raw_result["ids"]
    if ids is None:
        raise ValueError("ids must not be None")
    return [str(item) for item in ids]


def _distance_to_relevance_score(distance: float) -> float:
    if distance < 0.0:
        raise ValueError(f"distance must be non-negative, got: {distance}")
    return 1.0 / (1.0 + distance)
