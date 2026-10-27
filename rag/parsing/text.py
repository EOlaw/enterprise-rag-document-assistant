"""Plain-text and Markdown parsing.

Markdown support is a deliberately small line-based subset (no third-party parser):
ATX and setext headings, fenced code, pipe tables, lists, YAML front-matter titles and
HTML comments. Inline markup is reduced to its visible text so that BM25 and the
embedding model see what a human reader sees.
"""

from __future__ import annotations

import re

from rag.errors import UnsupportedEncodingError
from rag.parsing.models import BlockKind, RawBlock, RawDocument

_FORM_FEED = "\f"
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Only "**" is stripped: "__" also appears inside identifiers such as __init__.
_EMPHASIS = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_FRONT_MATTER_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)


def decode_text(data: bytes) -> str:
    """Decode UTF-8 (with or without BOM) or BOM-marked UTF-16.

    Legacy single-byte encodings are rejected rather than guessed: a wrong guess produces
    mojibake that is silently indexed and cited.
    """
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16")
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedEncodingError(
            "text is not valid UTF-8 or BOM-marked UTF-16; convert the file to UTF-8"
        ) from exc


def parse_plain_text(data: bytes) -> RawDocument:
    """Paragraphs split on blank lines; form feeds (``\\f``) mark page breaks when present."""
    text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
    paged = _FORM_FEED in text
    blocks: list[RawBlock] = []
    pages = text.split(_FORM_FEED)
    for page_number, page_text in enumerate(pages, start=1):
        for paragraph in re.split(r"\n[ \t]*\n", page_text):
            if paragraph.strip():
                blocks.append(
                    RawBlock(BlockKind.PARAGRAPH, paragraph, page=page_number if paged else None)
                )
    return RawDocument(blocks=blocks, page_count=len(pages) if paged else None)


def strip_inline_markdown(text: str) -> str:
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\1", text)
    return _INLINE_CODE.sub(r"\1", text)


def parse_markdown(data: bytes) -> RawDocument:
    text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
    text, front_matter_title = _split_front_matter(text)
    text = _HTML_COMMENT.sub("", text)

    blocks: list[RawBlock] = []
    buffer: list[str] = []
    buffer_kind = BlockKind.PARAGRAPH

    def flush() -> None:
        nonlocal buffer_kind
        if buffer:
            body = "\n".join(buffer)
            if buffer_kind is not BlockKind.CODE:
                body = strip_inline_markdown(body)
            blocks.append(RawBlock(buffer_kind, body))
            buffer.clear()
        buffer_kind = BlockKind.PARAGRAPH

    lines = text.split("\n")
    fence: str | None = None
    for line in lines:
        if fence is not None:
            if line.strip().startswith(fence):
                flush()
                fence = None
            else:
                buffer.append(line)
            continue

        fence_match = _FENCE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(1)
            buffer_kind = BlockKind.CODE
            continue

        if not line.strip():
            flush()
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            blocks.append(
                RawBlock(BlockKind.HEADING, strip_inline_markdown(heading.group(2)), level=level)
            )
            continue

        underline = _SETEXT_UNDERLINE.match(line)
        if underline and buffer_kind is BlockKind.PARAGRAPH and len(buffer) == 1:
            level = 1 if underline.group(1).startswith("=") else 2
            blocks.append(
                RawBlock(BlockKind.HEADING, strip_inline_markdown(buffer[0]), level=level)
            )
            buffer.clear()
            continue
        if underline:
            flush()  # thematic break ("---") or an unsupported multi-line setext heading
            continue

        stripped = line.strip()
        if stripped.startswith("|"):
            if buffer_kind is not BlockKind.TABLE:
                flush()
                buffer_kind = BlockKind.TABLE
            if not _TABLE_SEPARATOR.match(stripped):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                buffer.append(" | ".join(cells))
            continue

        if _LIST_ITEM.match(line):
            if buffer_kind is not BlockKind.LIST:
                flush()
                buffer_kind = BlockKind.LIST
            buffer.append(_LIST_ITEM.sub("- ", line, count=1).strip())
            continue

        if buffer_kind is BlockKind.LIST and buffer and line.startswith((" ", "\t")):
            buffer[-1] += " " + stripped  # continuation of the previous list item
            continue
        if buffer_kind is not BlockKind.PARAGRAPH:
            flush()
        buffer.append(line)
    flush()

    title = front_matter_title or next(
        (b.text.strip() for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), None
    )
    return RawDocument(blocks=blocks, title=title)


def _split_front_matter(text: str) -> tuple[str, str | None]:
    if not text.startswith("---\n"):
        return text, None
    end = text.find("\n---", 4)
    if end == -1:
        return text, None
    header = text[4:end]
    body_start = text.find("\n", end + 4)
    body = text[body_start + 1 :] if body_start != -1 else ""
    match = _FRONT_MATTER_TITLE.search(header)
    return body, match.group(1).strip() if match else None
