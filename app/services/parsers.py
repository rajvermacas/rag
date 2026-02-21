"""File parsing services for supported document formats."""

from pathlib import Path
import logging


logger = logging.getLogger(__name__)

TEXT_PLAIN = "text/plain"
APPLICATION_PDF = "application/pdf"
APPLICATION_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SUPPORTED_CONTENT_TYPES = {TEXT_PLAIN, APPLICATION_PDF, APPLICATION_DOCX}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file type is not supported by the parser."""


class EmptyExtractionError(ValueError):
    """Raised when text extraction returns no usable content."""


class ParserDependencyError(RuntimeError):
    """Raised when a required parser dependency is not installed."""


def parse_text_file(path: Path, content_type: str) -> str:
    """Parse text from a supported file type and fail on empty output."""
    logger.info("parse_text_file_started path=%s content_type=%s", path, content_type)
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise UnsupportedFileTypeError(f"Unsupported file type: {content_type}")

    if content_type == TEXT_PLAIN:
        extracted_text = _parse_txt(path)
    elif content_type == APPLICATION_PDF:
        extracted_text = _parse_pdf(path)
    else:
        extracted_text = _parse_docx(path)

    if extracted_text.strip() == "":
        raise EmptyExtractionError(f"No extractable text found in {path.name}")

    logger.info(
        "parse_text_file_completed path=%s content_type=%s char_count=%s",
        path,
        content_type,
        len(extracted_text),
    )
    return extracted_text


def _parse_txt(path: Path) -> str:
    logger.info("parse_txt_started path=%s", path)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"TXT parsing failed for {path.name}: invalid UTF-8 input") from exc
    logger.info("parse_txt_completed path=%s char_count=%s", path, len(content))
    return content


def _parse_pdf(path: Path) -> str:
    logger.info("parse_pdf_started path=%s", path)
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ParserDependencyError("Missing dependency for PDF parsing: pypdf") from exc

    pdf_reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text is not None:
            page_texts.append(text)

    combined_text = "\n".join(page_texts)
    logger.info("parse_pdf_completed path=%s char_count=%s", path, len(combined_text))
    return combined_text


def _parse_docx(path: Path) -> str:
    logger.info("parse_docx_started path=%s", path)
    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise ParserDependencyError("Missing dependency for DOCX parsing: python-docx") from exc

    document = Document(str(path))
    paragraph_texts = [paragraph.text for paragraph in document.paragraphs]
    combined_text = "\n".join(paragraph_texts)
    logger.info("parse_docx_completed path=%s char_count=%s", path, len(combined_text))
    return combined_text
