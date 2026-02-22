"""LlamaIndex ingestion and Chroma-backed document inventory service."""

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from uuid import uuid4

from app.services.vector_store import IndexedDocument


logger = logging.getLogger(__name__)
_METADATA_DOC_ID_KEY = "app_doc_id"
_METADATA_FILENAME_KEY = "filename"
_ALLOWED_UPLOAD_TYPES = MappingProxyType(
    {
        "text/plain": ".txt",
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }
)


class UploadLike(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self) -> bytes:
        """Read uploaded bytes."""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    chunks_indexed: int


class IndexingService:
    """Indexes upload content and manages indexed-document metadata."""

    def __init__(
        self,
        collection: Any,
        max_upload_mb: int,
        chunk_size: int,
        chunk_overlap: int,
        embed_model: Any,
    ) -> None:
        if collection is None:
            raise ValueError("collection must not be None")
        if max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be greater than 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if embed_model is None:
            raise ValueError("embed_model must not be None")
        self._collection = collection
        self._max_upload_bytes = max_upload_mb * 1024 * 1024
        self._pipeline = _build_ingestion_pipeline(
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model,
        )

    async def ingest_upload(self, upload: UploadLike) -> IngestResult:
        filename = _require_filename(upload.filename)
        content_type = _require_content_type(upload.content_type)
        file_bytes = await upload.read()
        _validate_upload_bytes(file_bytes, filename, self._max_upload_bytes)
        doc_id = str(uuid4())
        logger.info(
            "indexing_ingest_started filename=%s content_type=%s byte_count=%s doc_id=%s",
            filename,
            content_type,
            len(file_bytes),
            doc_id,
        )
        temp_path = _write_temp_file(file_bytes, filename)
        try:
            documents = _load_documents(temp_path)
            nodes = await self._pipeline.arun(
                documents=_attach_metadata(documents, doc_id, filename),
            )
        finally:
            _delete_temp_file_if_present(temp_path)
        if len(nodes) == 0:
            raise ValueError(f"ingestion produced no nodes for upload: {filename}")
        logger.info(
            "indexing_ingest_completed filename=%s doc_id=%s chunk_count=%s",
            filename,
            doc_id,
            len(nodes),
        )
        return IngestResult(doc_id=doc_id, chunks_indexed=len(nodes))

    def list_documents(self) -> list[IndexedDocument]:
        logger.info("indexing_list_documents_started")
        raw_result = self._collection.get(include=["metadatas"])
        documents = _build_indexed_documents(raw_result)
        logger.info("indexing_list_documents_completed document_count=%s", len(documents))
        return documents

    def delete_document(self, doc_id: str) -> int:
        normalized_doc_id = _require_non_empty_string(doc_id, "doc_id")
        logger.info("indexing_delete_document_started doc_id=%s", normalized_doc_id)
        raw_result = self._collection.get(where={_METADATA_DOC_ID_KEY: normalized_doc_id})
        ids = _extract_ids(raw_result)
        if len(ids) == 0:
            raise ValueError(f"document not found: {normalized_doc_id}")
        self._collection.delete(ids=ids)
        logger.info(
            "indexing_delete_document_completed doc_id=%s chunks_deleted=%s",
            normalized_doc_id,
            len(ids),
        )
        return len(ids)


def _build_ingestion_pipeline(
    collection: Any,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: Any,
) -> Any:
    try:
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for indexing service: llama-index") from exc
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            embed_model,
        ],
        vector_store=ChromaVectorStore(chroma_collection=collection),
    )


def _load_documents(temp_path: Path) -> list[Any]:
    try:
        from llama_index.core import SimpleDirectoryReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency for indexing service: llama-index") from exc
    documents = SimpleDirectoryReader(input_files=[str(temp_path)]).load_data()
    if len(documents) == 0:
        raise ValueError(f"No textual content extracted from upload: {temp_path.name}")
    return documents


def _attach_metadata(documents: list[Any], doc_id: str, filename: str) -> list[Any]:
    for document in documents:
        metadata = dict(document.metadata)
        metadata[_METADATA_DOC_ID_KEY] = doc_id
        metadata[_METADATA_FILENAME_KEY] = filename
        document.metadata = metadata
    return documents


def _build_indexed_documents(raw_result: Mapping[str, Any]) -> list[IndexedDocument]:
    metadatas = _extract_metadatas(raw_result)
    summary_by_doc_id: dict[str, IndexedDocument] = {}
    for metadata in metadatas:
        doc_id = _extract_required_metadata(metadata, _METADATA_DOC_ID_KEY)
        filename = _extract_required_metadata(metadata, _METADATA_FILENAME_KEY)
        summary_by_doc_id[doc_id] = _upsert_summary(summary_by_doc_id, doc_id, filename)
    return sorted(
        summary_by_doc_id.values(),
        key=lambda document: (document.filename.lower(), document.doc_id),
    )


def _upsert_summary(
    summary_by_doc_id: Mapping[str, IndexedDocument],
    doc_id: str,
    filename: str,
) -> IndexedDocument:
    existing = summary_by_doc_id.get(doc_id)
    if existing is None:
        return IndexedDocument(doc_id=doc_id, filename=filename, chunks_indexed=1)
    if existing.filename != filename:
        raise ValueError(
            f"inconsistent filename for doc_id={doc_id}: {existing.filename} != {filename}"
        )
    return IndexedDocument(
        doc_id=existing.doc_id,
        filename=existing.filename,
        chunks_indexed=existing.chunks_indexed + 1,
    )


def _extract_required_metadata(metadata: Mapping[str, Any], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"metadata entry is missing required key: {key}")
    value = metadata[key]
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"metadata value must be a non-empty string for key: {key}")
    return value


def _extract_metadatas(raw_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "metadatas" not in raw_result:
        raise ValueError("missing key in Chroma result: metadatas")
    metadatas = raw_result["metadatas"]
    if metadatas is None:
        raise ValueError("metadatas must not be None")
    return [metadata for metadata in metadatas if metadata is not None]


def _extract_ids(raw_result: Mapping[str, Any]) -> list[str]:
    if "ids" not in raw_result:
        raise ValueError("missing key in Chroma result: ids")
    ids = raw_result["ids"]
    if ids is None:
        raise ValueError("ids must not be None")
    return [str(item) for item in ids]


def _require_filename(filename: str | None) -> str:
    normalized = _require_non_empty_string(filename, "upload filename")
    if Path(normalized).suffix.strip() == "":
        raise ValueError("upload filename must include a file extension")
    return normalized


def _require_content_type(content_type: str | None) -> str:
    normalized = _require_non_empty_string(content_type, "upload content_type")
    if normalized not in _ALLOWED_UPLOAD_TYPES:
        supported_types = ", ".join(_ALLOWED_UPLOAD_TYPES.keys())
        raise ValueError(
            f"upload content_type is not supported: {normalized}; "
            f"supported types: {supported_types}"
        )
    return normalized


def _validate_upload_bytes(file_bytes: bytes, filename: str, max_upload_bytes: int) -> None:
    if len(file_bytes) == 0:
        raise ValueError(f"Uploaded file is empty: {filename}")
    if len(file_bytes) > max_upload_bytes:
        raise ValueError(
            f"Uploaded file exceeds max size {max_upload_bytes} bytes: {filename}"
        )


def _write_temp_file(file_bytes: bytes, filename: str) -> Path:
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        return Path(temp_file.name)


def _delete_temp_file_if_present(path: Path) -> None:
    if path.exists():
        path.unlink()


def _require_non_empty_string(value: str | None, field_name: str) -> str:
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} must be provided")
    return value
