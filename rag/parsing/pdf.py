"""PDF text extraction with page tracking (pypdf, BSD-licensed).

Limitations, stated rather than hidden:

* No OCR: image-only pages produce no text and a warning; an all-image PDF fails with
  ``empty_document``.
* No font-size heuristics, so PDF headings are not detected; chunks rely on page and
  paragraph boundaries instead.
* Page numbers are *physical* 1-based indices (what a PDF viewer navigates to), which can
  differ from printed page labels.
"""

from __future__ import annotations

import io
import logging
import math
import re

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from rag.config import IngestionLimits
from rag.errors import (
    CorruptedDocumentError,
    DocumentLimitExceededError,
    EncryptedDocumentError,
)
from rag.parsing.models import BlockKind, RawBlock, RawDocument

logger = logging.getLogger(__name__)

_MARGIN_LINES = 2  # lines inspected at the top and bottom of each page
_MARGIN_MIN_PAGES = 3
_MARGIN_PAGE_FRACTION = 0.6
_DIGITS = re.compile(r"\d+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def parse_pdf(data: bytes, limits: IngestionLimits) -> RawDocument:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            _decrypt_or_raise(reader)
        page_count = len(reader.pages)
    except (EncryptedDocumentError, DocumentLimitExceededError):
        raise
    except FileNotDecryptedError as exc:
        raise EncryptedDocumentError("PDF is password protected") from exc
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise CorruptedDocumentError(f"PDF could not be read: {type(exc).__name__}") from exc

    if page_count > limits.max_pdf_pages:
        raise DocumentLimitExceededError(
            f"PDF has {page_count} pages (limit {limits.max_pdf_pages})"
        )

    pages_lines: list[list[str]] = []
    failed_pages: list[int] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf raises many types on malformed content streams
            logger.warning(
                "pdf page extraction failed", extra={"page": index + 1, "error": repr(exc)}
            )
            failed_pages.append(index + 1)
            text = ""
        pages_lines.append(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

    pages_lines, removed = strip_repeated_margins(pages_lines)

    blocks: list[RawBlock] = []
    empty_pages: list[int] = []
    for page_number, lines in enumerate(pages_lines, start=1):
        page_text = "\n".join(lines)
        if not page_text.strip():
            empty_pages.append(page_number)
            continue
        for paragraph in _PARAGRAPH_BREAK.split(page_text):
            if paragraph.strip():
                blocks.append(RawBlock(BlockKind.PARAGRAPH, paragraph, page=page_number))

    warnings: list[str] = []
    if failed_pages:
        warnings.append(f"text extraction failed on pages {_compact(failed_pages)}")
    textless = [p for p in empty_pages if p not in failed_pages]
    if textless:
        warnings.append(
            f"no extractable text on pages {_compact(textless)} "
            "(possibly scanned images; OCR is not enabled)"
        )
    if removed:
        warnings.append(f"removed {removed} repeated header/footer lines")

    return RawDocument(
        blocks=blocks, title=_metadata_title(reader), page_count=page_count, warnings=warnings
    )


def _decrypt_or_raise(reader: PdfReader) -> None:
    # Many PDFs are "encrypted" with an empty user password purely to set permission flags;
    # those are readable by any viewer, so we read them too. Anything else is rejected.
    try:
        result = reader.decrypt("")
    except (FileNotDecryptedError, NotImplementedError, PdfReadError) as exc:
        raise EncryptedDocumentError(
            "PDF is encrypted with an unsupported or unknown password"
        ) from exc
    if not result:
        raise EncryptedDocumentError("PDF is password protected")


def strip_repeated_margins(pages_lines: list[list[str]]) -> tuple[list[list[str]], int]:
    """Remove running headers/footers: lines repeated in the margins of most pages.

    Digits are masked before comparison so "Page 3 of 28" matches "Page 4 of 28".
    Only the first/last ``_MARGIN_LINES`` non-blank lines of each page are candidates, so
    repeated text in the middle of a page is never removed. Known false positive: on very
    short pages every line is a margin line, so a body line that repeats (after digit
    masking) on most pages is removed as well; the count is reported as a warning.
    """
    pages_with_text = [lines for lines in pages_lines if any(line.strip() for line in lines)]
    if len(pages_with_text) < _MARGIN_MIN_PAGES:
        return pages_lines, 0

    def key(line: str) -> str:
        return _DIGITS.sub("#", " ".join(line.lower().split()))

    def margin_positions(lines: list[str]) -> list[int]:
        nonblank = [i for i, line in enumerate(lines) if line.strip()]
        return sorted(set(nonblank[:_MARGIN_LINES] + nonblank[-_MARGIN_LINES:]))

    counts: dict[str, int] = {}
    for lines in pages_with_text:
        for k in {key(lines[i]) for i in margin_positions(lines)}:
            counts[k] = counts.get(k, 0) + 1
    threshold = max(_MARGIN_MIN_PAGES, math.ceil(_MARGIN_PAGE_FRACTION * len(pages_with_text)))
    repeated = {k for k, c in counts.items() if c >= threshold}
    if not repeated:
        return pages_lines, 0

    removed = 0
    cleaned: list[list[str]] = []
    for lines in pages_lines:
        drop = {i for i in margin_positions(lines) if key(lines[i]) in repeated}
        removed += len(drop)
        cleaned.append([line for i, line in enumerate(lines) if i not in drop])
    return cleaned, removed


def _metadata_title(reader: PdfReader) -> str | None:
    try:
        title = reader.metadata.title if reader.metadata else None
    except Exception:  # malformed info dictionaries are common and non-fatal
        return None
    if isinstance(title, str) and title.strip():
        return " ".join(title.split())
    return None


def _compact(pages: list[int], limit: int = 20) -> str:
    shown = ", ".join(str(p) for p in pages[:limit])
    return shown + (f" (+{len(pages) - limit} more)" if len(pages) > limit else "")
