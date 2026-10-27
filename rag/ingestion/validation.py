"""Validation of untrusted uploads before any parser touches them.

The declared extension and the actual bytes must agree. We never trust a client-supplied
MIME type, and the original filename is display metadata only (storage keys are IDs).
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath

from rag.config import IngestionLimits
from rag.errors import (
    ContentTypeMismatchError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsafeArchiveError,
    UnsupportedFileTypeError,
)


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"

    @property
    def mime_type(self) -> str:
        return _MIME_TYPES[self]


_MIME_TYPES: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileType.TXT: "text/plain",
    FileType.MARKDOWN: "text/markdown",
}

_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
    ".text": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
}

# The PDF spec tolerates up to 1024 bytes of junk before the header.
_PDF_HEADER_WINDOW = 1024
_ZIP_MAGIC = b"PK\x03\x04"
_DOCX_REQUIRED_PARTS = ("[Content_Types].xml", "word/document.xml")
_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*]")
_MAX_FILENAME_CHARS = 255


def safe_display_name(filename: str) -> str:
    """Reduce a client-supplied filename to a safe display name.

    Strips directory components (both separators), control and reserved characters,
    and bounds the length while keeping the extension.
    """
    name = unicodedata.normalize("NFC", filename)
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name:
        return "untitled"
    if len(name) > _MAX_FILENAME_CHARS:
        suffix = PurePosixPath(name).suffix[:16]
        name = name[: _MAX_FILENAME_CHARS - len(suffix)] + suffix
    return name


def detect_file_type(filename: str, data: bytes, limits: IngestionLimits) -> FileType:
    """Return the validated file type or raise a typed :class:`IngestionError`."""
    if not data:
        raise EmptyDocumentError("uploaded file is empty")
    if len(data) > limits.max_file_bytes:
        raise FileTooLargeError(
            f"file is {len(data)} bytes; the limit is {limits.max_file_bytes} bytes"
        )

    extension = PurePosixPath(filename.lower()).suffix
    claimed = _EXTENSIONS.get(extension)
    if claimed is None:
        supported = ", ".join(sorted(_EXTENSIONS))
        raise UnsupportedFileTypeError(f"extension {extension!r} is not supported ({supported})")

    if claimed is FileType.PDF:
        if b"%PDF-" not in data[:_PDF_HEADER_WINDOW]:
            raise ContentTypeMismatchError("file has a .pdf extension but no PDF header")
    elif claimed is FileType.DOCX:
        if not data.startswith(_ZIP_MAGIC):
            raise ContentTypeMismatchError("file has a .docx extension but is not a ZIP container")
        check_docx_container(data, limits)
    elif b"\x00" in data and not _has_utf16_bom(data):
        raise ContentTypeMismatchError("text file contains binary (NUL) bytes")
    return claimed


def check_docx_container(data: bytes, limits: IngestionLimits) -> None:
    """Reject zip bombs, macro-enabled documents and non-Word ZIP files.

    Uses the central directory only (no decompression), so it is cheap and safe to run on
    hostile input. Declared sizes can lie; the parser's extraction-size limit is the
    second line of defence.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, ValueError) as exc:
        raise UnsafeArchiveError("DOCX container is not a readable ZIP archive") from exc

    if len(infos) > limits.max_zip_entries:
        raise UnsafeArchiveError(
            f"DOCX contains {len(infos)} entries (limit {limits.max_zip_entries})"
        )

    names = {info.filename for info in infos}
    missing = [part for part in _DOCX_REQUIRED_PARTS if part not in names]
    if missing:
        raise ContentTypeMismatchError(f"ZIP file is not a Word document (missing {missing})")
    if any(name.lower().endswith("vbaproject.bin") for name in names):
        raise UnsafeArchiveError("macro-enabled documents are not accepted")

    total_uncompressed = 0
    for info in infos:
        total_uncompressed += info.file_size
        if info.compress_size > 0 and info.file_size / info.compress_size > (
            limits.max_zip_compression_ratio
        ):
            raise UnsafeArchiveError(
                f"entry {info.filename!r} exceeds the maximum compression ratio"
            )
    if total_uncompressed > limits.max_zip_uncompressed_bytes:
        raise UnsafeArchiveError(
            f"DOCX expands to {total_uncompressed} bytes "
            f"(limit {limits.max_zip_uncompressed_bytes})"
        )


def _has_utf16_bom(data: bytes) -> bool:
    return data.startswith((b"\xff\xfe", b"\xfe\xff"))
