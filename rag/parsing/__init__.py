"""Format dispatch for document parsers."""

from __future__ import annotations

from rag.config import IngestionLimits
from rag.ingestion.validation import FileType
from rag.parsing.docx import parse_docx
from rag.parsing.models import RawDocument
from rag.parsing.pdf import parse_pdf
from rag.parsing.text import parse_markdown, parse_plain_text

# Bump whenever parser or normalization output can change for the same input bytes.
# It is part of the index fingerprint, so a bump makes existing indexes visibly stale.
PARSER_VERSION = "1"


def parse_document(file_type: FileType, data: bytes, limits: IngestionLimits) -> RawDocument:
    match file_type:
        case FileType.PDF:
            return parse_pdf(data, limits)
        case FileType.DOCX:
            return parse_docx(data)
        case FileType.TXT:
            return parse_plain_text(data)
        case FileType.MARKDOWN:
            return parse_markdown(data)
