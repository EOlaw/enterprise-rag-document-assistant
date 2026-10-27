"""Text normalization and canonical-text assembly.

Design choices (each is a trade-off, documented in docs/rag/retrieval-design.md):

* NFKC folds ligatures, full-width forms and non-breaking spaces that PDF extraction emits
  ("\ufb01" -> "fi"). It also folds superscripts ("x\u00b2" -> "x2"), which we accept.
* Hyphens at line ends are *kept* and the lines are joined without a space. Removing them
  ("manage-\\nment" -> "management") would corrupt compounds and identifiers such as
  "self-service" or "AC-2"; keeping them costs a little lexical recall on true hyphenation.
* Bullet and numbered-list lines keep their own line so list structure survives.
"""

from __future__ import annotations

import re
import unicodedata

from rag.parsing.models import Block, BlockKind, RawBlock

BLOCK_SEPARATOR = "\n\n"

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HORIZONTAL_SPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
# Symbol/Wingdings fonts emit Private Use Area code points (e.g. U+F03C, U+F0B7) as bullets.
_PUA_BULLET = re.compile(r"^[-](?=\s)", re.MULTILINE)
_PUA = re.compile(r"[-]")
_LIST_MARKER = re.compile(
    r"^(?:[-*+\u2022\u25aa\u25e6\u2023\u2013]|\(?\d{1,3}[.)]|\(?[a-z][.)])\s+"
)


def clean_text(text: str) -> str:
    """Unicode-normalize, drop invisible/control characters and collapse horizontal space.

    Line structure is preserved; callers decide whether lines are joined.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_INVISIBLE)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _PUA.sub("", _PUA_BULLET.sub("•", text))
    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(lines).strip("\n")


def join_wrapped_lines(text: str) -> str:
    """Join hard-wrapped lines into flowing text while keeping list items on their own line."""
    out: list[str] = []
    for line in text.split("\n"):
        if not line:
            continue
        if not out or _LIST_MARKER.match(line):
            out.append(line)
        elif out[-1].endswith("-"):
            out[-1] += line  # keep the hyphen: see module docstring
        else:
            out[-1] += " " + line
    return "\n".join(out)


def normalize_block_text(kind: BlockKind, text: str) -> str:
    if kind is BlockKind.CODE:
        # Keep indentation; only normalize newlines and strip invisible characters.
        code = text.replace("\r\n", "\n").replace("\r", "\n").translate(_INVISIBLE)
        return _CONTROL_CHARS.sub(" ", code).strip("\n")
    cleaned = clean_text(text)
    if kind.preserves_lines:
        return "\n".join(line for line in cleaned.split("\n") if line)
    return join_wrapped_lines(cleaned) if kind is BlockKind.PARAGRAPH else " ".join(cleaned.split())


def build_canonical_text(raw_blocks: list[RawBlock]) -> tuple[str, list[Block]]:
    """Normalize blocks and lay them out in one canonical string with exact offsets.

    Blocks that normalize to empty text are dropped. The invariant
    ``canonical[b.start:b.end] == b.text`` holds for every returned block.
    """
    parts: list[str] = []
    blocks: list[Block] = []
    cursor = 0
    for raw in raw_blocks:
        text = normalize_block_text(raw.kind, raw.text)
        if not text.strip():
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        start = cursor
        parts.append(text)
        cursor += len(text)
        blocks.append(Block(raw.kind, text, start, cursor, raw.page, raw.level))
    return "".join(parts), blocks
