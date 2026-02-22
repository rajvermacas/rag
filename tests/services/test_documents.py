from dataclasses import dataclass

import pytest

from app.services.documents import DocumentService


@dataclass(frozen=True)
class FakeIndexedDocument:
    doc_id: str
    filename: str
    chunks_indexed: int


class FakeDocumentStore:
    def __init__(self) -> None:
        self.documents = [
            FakeIndexedDocument(doc_id="doc-1", filename="alpha.txt", chunks_indexed=2),
            FakeIndexedDocument(doc_id="doc-2", filename="beta.txt", chunks_indexed=1),
        ]
        self.deleted_chunk_count = 1

    def list_documents(self) -> list[FakeIndexedDocument]:
        return list(self.documents)

    def delete_document(self, doc_id: str) -> int:
        if doc_id == "missing":
            return 0
        return self.deleted_chunk_count


def test_list_documents_returns_summaries() -> None:
    service = DocumentService(vector_store=FakeDocumentStore())

    documents = service.list_documents()

    assert len(documents) == 2
    assert documents[0].doc_id == "doc-1"
    assert documents[1].filename == "beta.txt"


def test_delete_document_raises_when_store_reports_no_chunks_deleted() -> None:
    service = DocumentService(vector_store=FakeDocumentStore())

    with pytest.raises(ValueError, match="document not found: missing"):
        service.delete_document("missing")
