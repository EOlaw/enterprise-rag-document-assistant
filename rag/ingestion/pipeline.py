"""The pure document-processing pipeline: bytes in, citeable chunks out.

No I/O happens here. The ingestion worker (Phase 2) wraps this function with object
storage and PostgreSQL state transitions. Vector indexing follows in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from rag.chunking.chunker import Chunk, chunk_document
from rag.chunking.tokens import TokenCounter
from rag.config import ChunkingConfig, IngestionLimits
from rag.errors import DocumentLimitExceededError, EmptyDocumentError
from rag.ingestion.identity import sha256_hex, text_fingerprint
from rag.ingestion.validation import FileType, detect_file_type, safe_display_name
from rag.parsing import PARSER_VERSION, parse_document
from rag.parsing.models import Block
from rag.parsing.normalize import build_canonical_text


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    filename: str
    file_type: FileType
    byte_size: int
    raw_sha256: str
    text_sha256: str
    title: str
    page_count: int | None
    canonical_text: str
    blocks: list[Block]
    chunks: list[Chunk]
    parser_version: str
    chunking_fingerprint: str
    tokenizer: str
    warnings: list[str] = field(default_factory=list)

    @property
    def mime_type(self) -> str:
        return self.file_type.mime_type


def process_document(
    data: bytes,
    filename: str,
    *,
    chunking: ChunkingConfig,
    token_counter: TokenCounter,
    limits: IngestionLimits | None = None,
) -> ProcessedDocument:
    """Validate, parse, normalize and chunk one uploaded file.

    Raises a subclass of :class:`rag.errors.IngestionError` for any input problem.
    """
    limits = limits or IngestionLimits()
    display_name = safe_display_name(filename)
    file_type = detect_file_type(display_name, data, limits)

    raw = parse_document(file_type, data, limits)
    canonical_text, blocks = build_canonical_text(raw.blocks)
    if not canonical_text.strip():
        raise EmptyDocumentError(
            "no extractable text (scanned documents require OCR, which is not enabled)"
        )
    if len(canonical_text) > limits.max_extracted_chars:
        raise DocumentLimitExceededError(
            f"extracted text has {len(canonical_text)} characters "
            f"(limit {limits.max_extracted_chars})"
        )

    chunks = chunk_document(canonical_text, blocks, chunking, token_counter)
    if not chunks:
        raise EmptyDocumentError("document contains headings but no indexable body text")

    return ProcessedDocument(
        filename=display_name,
        file_type=file_type,
        byte_size=len(data),
        raw_sha256=sha256_hex(data),
        text_sha256=text_fingerprint(canonical_text),
        title=raw.title or PurePosixPath(display_name).stem,
        page_count=raw.page_count,
        canonical_text=canonical_text,
        blocks=blocks,
        chunks=chunks,
        parser_version=PARSER_VERSION,
        chunking_fingerprint=chunking.fingerprint(),
        tokenizer=token_counter.name,
        warnings=list(raw.warnings),
    )
