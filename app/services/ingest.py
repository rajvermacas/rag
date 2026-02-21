"""Document ingestion workflow for upload, parsing, chunking, and indexing."""

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from typing import Protocol
from uuid import uuid4

from app.services.chunking import chunk_text
from app.services.parsers import parse_text_file


logger = logging.getLogger(__name__)


class UploadLike(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self) -> bytes:
        """Read upload bytes."""


class EmbedClient(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings."""


class VectorStore(Protocol):
    def upsert_chunks(
        self, doc_id: str, filename: str, chunks: list[str], embeddings: list[list[float]]
    ) -> int:
        """Upsert chunks and embeddings into vector store."""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    chunks_indexed: int


class IngestService:
    """Service that ingests uploads into the vector store."""

    def __init__(
        self,
        embed_client: EmbedClient,
        vector_store: VectorStore,
        max_upload_mb: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be greater than 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        self._embed_client = embed_client
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_mb * 1024 * 1024
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_upload(self, upload: UploadLike) -> IngestResult:
        if upload.filename is None or upload.filename.strip() == "":
            raise ValueError("upload filename must be provided")
        if upload.content_type is None or upload.content_type.strip() == "":
            raise ValueError("upload content_type must be provided")

        file_bytes = await upload.read()
        if len(file_bytes) == 0:
            raise ValueError(f"Uploaded file is empty: {upload.filename}")
        if len(file_bytes) > self._max_upload_bytes:
            raise ValueError(
                f"Uploaded file exceeds max size {self._max_upload_bytes} bytes: "
                f"{upload.filename}"
            )

        logger.info(
            "ingest_upload_started filename=%s content_type=%s byte_count=%s",
            upload.filename,
            upload.content_type,
            len(file_bytes),
        )
        temp_path = _write_temp_file(file_bytes, upload.filename)
        try:
            extracted_text = parse_text_file(temp_path, upload.content_type)
            chunks = chunk_text(extracted_text, self._chunk_size, self._chunk_overlap)
            embeddings = await self._embed_client.embed_texts(chunks)
            doc_id = str(uuid4())
            chunk_count = self._vector_store.upsert_chunks(
                doc_id=doc_id,
                filename=upload.filename,
                chunks=chunks,
                embeddings=embeddings,
            )
            logger.info(
                "ingest_upload_completed filename=%s doc_id=%s chunk_count=%s",
                upload.filename,
                doc_id,
                chunk_count,
            )
            return IngestResult(doc_id=doc_id, chunks_indexed=chunk_count)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _write_temp_file(file_bytes: bytes, filename: str) -> Path:
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        return Path(temp_file.name)
