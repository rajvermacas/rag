"""Document inventory and deletion service."""

from dataclasses import dataclass
import logging
from typing import Protocol

from app.services.vector_store import IndexedDocument


logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def list_documents(self) -> list[IndexedDocument]:
        """Return all indexed documents."""

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for the document and return deleted chunk count."""


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    filename: str
    chunks_indexed: int


class DocumentService:
    """Expose indexed-document operations for API/UI use."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def list_documents(self) -> list[DocumentSummary]:
        logger.info("document_service_list_started")
        documents = self._vector_store.list_documents()
        summaries = [
            DocumentSummary(
                doc_id=document.doc_id,
                filename=document.filename,
                chunks_indexed=document.chunks_indexed,
            )
            for document in documents
        ]
        logger.info("document_service_list_completed document_count=%s", len(summaries))
        return summaries

    def delete_document(self, doc_id: str) -> int:
        if doc_id.strip() == "":
            raise ValueError("doc_id must not be empty")
        logger.info("document_service_delete_started doc_id=%s", doc_id)
        deleted_chunks = self._vector_store.delete_document(doc_id)
        logger.info(
            "document_service_delete_completed doc_id=%s chunks_deleted=%s",
            doc_id,
            deleted_chunks,
        )
        return deleted_chunks
