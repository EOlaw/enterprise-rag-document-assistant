"""DOCX extraction in reading order: headings, paragraphs, lists and tables.

DOCX has no stable page model (pagination happens at render time), so blocks carry
``page=None`` and citations point at section headings and character offsets instead.
Headers, footers, comments and tracked-change deletions are not indexed.
"""

from __future__ import annotations

import io
import re
import zipfile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag.errors import CorruptedDocumentError
from rag.parsing.models import BlockKind, RawBlock, RawDocument

_HEADING_STYLE = re.compile(r"^heading\s+(\d)$", re.IGNORECASE)


def parse_docx(data: bytes) -> RawDocument:
    try:
        document = Document(io.BytesIO(data))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise CorruptedDocumentError(f"DOCX could not be read: {type(exc).__name__}") from exc
    except Exception as exc:  # lxml raises XMLSyntaxError for malformed parts
        raise CorruptedDocumentError(f"DOCX could not be parsed: {type(exc).__name__}") from exc

    blocks: list[RawBlock] = []
    list_buffer: list[str] = []

    def flush_list() -> None:
        if list_buffer:
            blocks.append(RawBlock(BlockKind.LIST, "\n".join(list_buffer)))
            list_buffer.clear()

    for item in document.iter_inner_content():
        if isinstance(item, Table):
            flush_list()
            table_text = _table_text(item)
            if table_text:
                blocks.append(RawBlock(BlockKind.TABLE, table_text))
            continue
        paragraph: Paragraph = item
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        level = _heading_level(style)
        if level is not None:
            flush_list()
            blocks.append(RawBlock(BlockKind.HEADING, text, level=level))
        elif "list" in style.lower():
            list_buffer.append(text)
        else:
            flush_list()
            blocks.append(RawBlock(BlockKind.PARAGRAPH, text))
    flush_list()

    title = (document.core_properties.title or "").strip() or None
    return RawDocument(blocks=blocks, title=title, page_count=None)


def _heading_level(style_name: str) -> int | None:
    if style_name.lower() == "title":
        return 1
    match = _HEADING_STYLE.match(style_name.strip())
    return int(match.group(1)) if match else None


def _table_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            # Horizontally merged cells are returned once per grid column; emit them once.
            if previous is not None and cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
