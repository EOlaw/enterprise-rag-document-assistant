from __future__ import annotations

import pytest

from rag.config import IngestionLimits
from rag.errors import (
    ContentTypeMismatchError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsafeArchiveError,
    UnsupportedFileTypeError,
)
from rag.ingestion.validation import FileType, detect_file_type, safe_display_name
from tests.builders import make_docx, make_pdf, make_zip

LIMITS = IngestionLimits()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.PDF", FileType.PDF),
        ("notes.md", FileType.MARKDOWN),
        ("a.markdown", FileType.MARKDOWN),
        ("readme.txt", FileType.TXT),
    ],
)
def test_accepts_supported_types(filename: str, expected: FileType) -> None:
    data = make_pdf([["hello"]]) if expected is FileType.PDF else b"# Title\n\nbody"
    assert detect_file_type(filename, data, LIMITS) is expected


def test_accepts_real_docx() -> None:
    data = make_docx([("paragraph", "hello")])
    assert detect_file_type("policy.docx", data, LIMITS) is FileType.DOCX


def test_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type("payload.exe", b"MZ\x90\x00", LIMITS)


def test_rejects_empty_file() -> None:
    with pytest.raises(EmptyDocumentError):
        detect_file_type("empty.txt", b"", LIMITS)


def test_rejects_oversized_file() -> None:
    with pytest.raises(FileTooLargeError):
        detect_file_type("big.txt", b"x" * 11, IngestionLimits(max_file_bytes=10))


def test_rejects_executable_renamed_to_pdf() -> None:
    with pytest.raises(ContentTypeMismatchError):
        detect_file_type("invoice.pdf", b"MZ\x90\x00" + b"\x00" * 100, LIMITS)


def test_rejects_binary_renamed_to_txt() -> None:
    with pytest.raises(ContentTypeMismatchError):
        detect_file_type("notes.txt", b"\x7fELF\x02\x01\x01\x00\x00", LIMITS)


def test_rejects_plain_zip_renamed_to_docx() -> None:
    with pytest.raises(ContentTypeMismatchError):
        detect_file_type("policy.docx", make_zip({"readme.txt": b"hi"}), LIMITS)


def test_rejects_macro_enabled_docx() -> None:
    data = make_zip(
        {
            "[Content_Types].xml": b"<Types/>",
            "word/document.xml": b"<w/>",
            "word/vbaProject.bin": b"\x00",
        }
    )
    with pytest.raises(UnsafeArchiveError, match="macro"):
        detect_file_type("policy.docx", data, LIMITS)


def test_rejects_zip_bomb_by_compression_ratio() -> None:
    data = make_zip({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"a" * 2_000_000})
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        detect_file_type("bomb.docx", data, IngestionLimits(max_zip_compression_ratio=50))


def test_rejects_docx_exceeding_uncompressed_limit() -> None:
    data = make_docx([("paragraph", "hello")])
    with pytest.raises(UnsafeArchiveError, match="expands to"):
        detect_file_type("policy.docx", data, IngestionLimits(max_zip_uncompressed_bytes=1000))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\victim\\secret.pdf", "secret.pdf"),
        ("report<script>.pdf", "report_script_.pdf"),
        ("name\x00.pdf", "name_.pdf"),
        ("   ", "untitled"),
    ],
)
def test_safe_display_name_strips_paths_and_reserved_characters(raw: str, expected: str) -> None:
    assert safe_display_name(raw) == expected


def test_safe_display_name_bounds_length_and_keeps_extension() -> None:
    name = safe_display_name("a" * 400 + ".pdf")
    assert len(name) == 255
    assert name.endswith(".pdf")
