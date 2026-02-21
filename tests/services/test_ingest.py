import asyncio

import pytest

from app.services.ingest import IngestService


class FakeUpload:
    def __init__(self, filename: str | None, content_type: str | None, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeEmbedClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_doc_id = ""
        self.last_filename = ""
        self.last_chunks: list[str] = []

    def upsert_chunks(
        self, doc_id: str, filename: str, chunks: list[str], embeddings: list[list[float]]
    ) -> int:
        if len(chunks) != len(embeddings):
            raise AssertionError("chunks and embeddings must have equal length")
        self.last_doc_id = doc_id
        self.last_filename = filename
        self.last_chunks = chunks
        return len(chunks)


def test_ingest_upload_txt_success() -> None:
    vector_store = FakeVectorStore()
    service = IngestService(
        embed_client=FakeEmbedClient(),
        vector_store=vector_store,
        max_upload_mb=1,
        chunk_size=5,
        chunk_overlap=1,
    )
    upload = FakeUpload(filename="a.txt", content_type="text/plain", data=b"hello world")

    result = asyncio.run(service.ingest_upload(upload))

    assert result.doc_id != ""
    assert result.chunks_indexed > 0
    assert vector_store.last_filename == "a.txt"


def test_ingest_upload_missing_content_type_raises() -> None:
    service = IngestService(
        embed_client=FakeEmbedClient(),
        vector_store=FakeVectorStore(),
        max_upload_mb=1,
        chunk_size=5,
        chunk_overlap=1,
    )
    upload = FakeUpload(filename="a.txt", content_type=None, data=b"hello")

    with pytest.raises(ValueError, match="upload content_type must be provided"):
        asyncio.run(service.ingest_upload(upload))


def test_ingest_upload_oversized_file_raises() -> None:
    service = IngestService(
        embed_client=FakeEmbedClient(),
        vector_store=FakeVectorStore(),
        max_upload_mb=1,
        chunk_size=5,
        chunk_overlap=1,
    )
    upload = FakeUpload(
        filename="a.txt",
        content_type="text/plain",
        data=b"a" * (1024 * 1024 + 1),
    )

    with pytest.raises(ValueError, match="Uploaded file exceeds max size"):
        asyncio.run(service.ingest_upload(upload))
