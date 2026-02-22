from pathlib import Path
import asyncio

import pytest

from app.services.indexing import IndexingService


class FakeUpload:
    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


def fake_upload(filename: str, content_type: str, data: bytes) -> FakeUpload:
    return FakeUpload(filename=filename, content_type=content_type, data=data)


def build_indexing_service(tmp_path: Path) -> IndexingService:
    import chromadb
    from llama_index.core.embeddings import MockEmbedding

    persist_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="rag_docs")
    return IndexingService(
        collection=collection,
        max_upload_mb=2,
        chunk_size=64,
        chunk_overlap=8,
        embed_model=MockEmbedding(embed_dim=16),
    )


def test_ingest_upload_indexes_document_and_returns_counts(tmp_path: Path) -> None:
    service = build_indexing_service(tmp_path)
    upload = fake_upload("notes.txt", "text/plain", b"hello world")

    result = asyncio.run(service.ingest_upload(upload))

    assert result.doc_id != ""
    assert result.chunks_indexed > 0


def test_delete_document_raises_when_doc_not_found(tmp_path: Path) -> None:
    service = build_indexing_service(tmp_path)

    with pytest.raises(ValueError, match="document not found"):
        service.delete_document("missing")
