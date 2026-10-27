"""Format-independent document representation.

Parsers emit :class:`RawBlock` objects; normalization turns them into :class:`Block`
objects positioned inside a single *canonical text*. All downstream offsets (chunks,
citations, highlight spans) refer to that canonical text, which is stored alongside each
document version so offsets remain verifiable forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"  # wrapped lines are joined into flowing text
    LIST = "list"  # one item per line, line breaks preserved
    TABLE = "table"  # one row per line, cells separated by " | "
    CODE = "code"  # verbatim, whitespace preserved

    @property
    def preserves_lines(self) -> bool:
        return self is not BlockKind.PARAGRAPH and self is not BlockKind.HEADING


@dataclass(frozen=True, slots=True)
class RawBlock:
    kind: BlockKind
    text: str
    page: int | None = None  # 1-based physical page number; None when the format has no pages
    level: int | None = None  # heading level (1 = top), only for HEADING blocks


@dataclass(frozen=True, slots=True)
class RawDocument:
    blocks: list[RawBlock]
    title: str | None = None
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Block:
    """A normalized block located at ``canonical_text[start:end]``."""

    kind: BlockKind
    text: str
    start: int
    end: int
    page: int | None = None
    level: int | None = None
