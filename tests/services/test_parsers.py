import pytest

from app.services.parsers import (
    EmptyExtractionError,
    UnsupportedFileTypeError,
    parse_text_file,
)


def test_txt_parser_returns_text(tmp_path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = parse_text_file(file_path, "text/plain")

    assert result == "hello"


def test_unsupported_file_type_raises(tmp_path) -> None:
    file_path = tmp_path / "a.csv"
    file_path.write_text("x,y", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        parse_text_file(file_path, "text/csv")


def test_empty_text_raises(tmp_path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyExtractionError):
        parse_text_file(file_path, "text/plain")
