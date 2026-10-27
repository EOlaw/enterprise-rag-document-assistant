"""Typed ingestion errors.

Every error carries a stable machine-readable ``code`` that is persisted on the ingestion
job record and shown in the UI, plus a ``retryable`` flag the job runner uses to decide
between retrying (transient infrastructure failures) and failing fast (bad input).
Messages must never contain document text: they end up in logs and job records.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for document ingestion failures."""

    code: str = "ingestion_error"
    retryable: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class UnsupportedFileTypeError(IngestionError):
    code = "unsupported_file_type"


class ContentTypeMismatchError(IngestionError):
    """The file extension claims one format but the bytes are another."""

    code = "content_type_mismatch"


class FileTooLargeError(IngestionError):
    code = "file_too_large"


class CorruptedDocumentError(IngestionError):
    code = "corrupted_document"


class EncryptedDocumentError(IngestionError):
    code = "encrypted_document"


class EmptyDocumentError(IngestionError):
    """No indexable text was extracted (e.g. a scanned PDF without OCR)."""

    code = "empty_document"


class UnsafeArchiveError(IngestionError):
    """A DOCX container looks like a zip bomb, contains macros, or is malformed."""

    code = "unsafe_archive"


class DocumentLimitExceededError(IngestionError):
    code = "limit_exceeded"


class UnsupportedEncodingError(IngestionError):
    code = "unsupported_encoding"


class ConfigurationError(ValueError):
    """Invalid pipeline configuration (a programming/deployment error, not bad input)."""
