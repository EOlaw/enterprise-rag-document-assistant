"""Chunking strategies.

Invariants (enforced by tests):

1. ``chunk.text == canonical_text[chunk.char_start:chunk.char_end]``: every chunk is an
   exact slice, so citations can be highlighted precisely.
2. ``token_count(chunk.embedding_text) <= config.max_tokens``: nothing is silently
   truncated by the embedding model.
3. All body text is covered by at least one chunk (headings are carried in the prefix).

``structure_aware`` packs whole blocks -> sentences -> token windows, never crossing a
section (or, optionally, page) boundary. ``fixed_tokens`` is the naive baseline used in
chunking experiments.
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

from rag.chunking.sentences import sentence_spans
from rag.chunking.tokens import Span, TokenCounter
from rag.config import ChunkingConfig
from rag.errors import ConfigurationError
from rag.parsing.models import Block, BlockKind

HEADING_SEPARATOR = " > "
PREFIX_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    embedding_text: str  # heading prefix + text; what is embedded and BM25-indexed
    token_count: int  # of embedding_text, in the configured tokenizer
    content_hash: str  # sha256(embedding_text): embedding cache key together with model id


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    page: int | None


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: tuple[Block, ...]


def chunk_document(
    canonical_text: str,
    blocks: list[Block],
    config: ChunkingConfig,
    counter: TokenCounter,
) -> list[Chunk]:
    limit = counter.max_input_tokens
    if limit is not None and config.max_tokens > limit:
        raise ConfigurationError(
            f"max_tokens={config.max_tokens} exceeds what {counter.name} embeds "
            f"({limit} tokens); chunks would be silently truncated"
        )
    if config.strategy == "fixed_tokens":
        return _fixed_token_chunks(canonical_text, blocks, config, counter)
    return _structure_aware_chunks(canonical_text, blocks, config, counter)


# --------------------------------------------------------------------------- structure-aware


def _structure_aware_chunks(
    text: str, blocks: list[Block], config: ChunkingConfig, counter: TokenCounter
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _sections(blocks, config):
        prefix = _heading_prefix(section.heading_path, config, counter)
        prefix_tokens = counter.count(prefix)
        budget = config.max_tokens - prefix_tokens
        units: list[_Unit] = []
        for block in section.blocks:
            units.extend(_block_units(text, block, budget, config.overlap_tokens, counter))
        for group in _pack(text, units, budget, config.overlap_tokens, counter):
            chunks.append(
                _make_chunk(len(chunks), text, group, section.heading_path, prefix, counter)
            )
    return chunks


def _sections(blocks: list[Block], config: ChunkingConfig) -> list[_Section]:
    sections: list[_Section] = []
    path: list[tuple[int, str]] = []  # (level, heading text)
    current: list[Block] = []

    def close() -> None:
        if current:
            sections.append(_Section(tuple(h for _, h in path), tuple(current)))
            current.clear()

    for block in blocks:
        if block.kind is BlockKind.HEADING and config.heading_aware:
            close()
            level = block.level or 1
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, block.text))
            continue  # heading text travels in the prefix, not in the body
        if config.page_boundaries and current and current[-1].page != block.page:
            close()
        current.append(block)
    close()
    return sections


def _heading_prefix(path: tuple[str, ...], config: ChunkingConfig, counter: TokenCounter) -> str:
    if not path or not config.include_heading_prefix or config.max_heading_prefix_tokens == 0:
        return ""
    parts = list(path)
    while parts:
        prefix = HEADING_SEPARATOR.join(parts)
        if counter.count(prefix) <= config.max_heading_prefix_tokens:
            return prefix + PREFIX_SEPARATOR
        if len(parts) == 1:
            break
        parts.pop(0)  # drop the outermost ancestor first; the nearest heading matters most
    spans = counter.spans(parts[0])[: config.max_heading_prefix_tokens]
    return parts[0][: spans[-1][1]] + PREFIX_SEPARATOR if spans else ""


def _block_units(
    text: str, block: Block, budget: int, overlap: int, counter: TokenCounter
) -> list[_Unit]:
    """Split a block into units that each fit the budget: whole block > sentences > tokens."""
    if counter.count(block.text) <= budget:
        return [_Unit(block.start, block.end, block.page)]
    units: list[_Unit] = []
    for s_start, s_end in sentence_spans(block.text):
        start, end = block.start + s_start, block.start + s_end
        if counter.count(text[start:end]) <= budget:
            units.append(_Unit(start, end, block.page))
        else:
            for w_start, w_end in _token_windows(text, start, end, budget, overlap, counter):
                units.append(_Unit(w_start, w_end, block.page))
    return units


def _token_windows(
    text: str, start: int, end: int, budget: int, overlap: int, counter: TokenCounter
) -> list[Span]:
    """Split ``text[start:end]`` into overlapping windows of at most ``budget`` tokens.

    Re-tokenizing a slice can differ from the token count inside the full string (for
    example when a cut lands inside a word), so each window is verified and shrunk.
    """
    segment = text[start:end]
    spans = counter.spans(segment)
    windows: list[Span] = []
    i = 0
    while i < len(spans):
        j = min(i + budget, len(spans))
        while j > i + 1 and counter.count(segment[spans[i][0] : spans[j - 1][1]]) > budget:
            j -= 1
        windows.append((start + spans[i][0], start + spans[j - 1][1]))
        if j >= len(spans):
            break
        i = max(j - overlap, i + 1)
    return windows


def _pack(
    text: str, units: list[_Unit], budget: int, overlap: int, counter: TokenCounter
) -> list[list[_Unit]]:
    """Greedily pack consecutive units into groups whose span fits the budget.

    Overlap is unit-granular: trailing whole units (sentences/blocks) of the previous
    group, up to ``overlap`` tokens, are repeated at the start of the next one.
    """
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and counter.count(text[current[0].start : unit.end]) > budget:
            groups.append(current)
            current = _overlap_tail(text, current, overlap, counter)
            while current and counter.count(text[current[0].start : unit.end]) > budget:
                current.pop(0)
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def _overlap_tail(
    text: str, units: list[_Unit], overlap: int, counter: TokenCounter
) -> list[_Unit]:
    if overlap == 0:
        return []
    tail: list[_Unit] = []
    for unit in reversed(units[1:]):  # never carry the whole group forward
        if counter.count(text[unit.start : units[-1].end]) > overlap:
            break
        tail.insert(0, unit)
    return tail


def _make_chunk(
    index: int,
    text: str,
    units: list[_Unit],
    heading_path: tuple[str, ...],
    prefix: str,
    counter: TokenCounter,
) -> Chunk:
    pages = [u.page for u in units if u.page is not None]
    return _chunk(
        index=index,
        text=text,
        span=(units[0].start, max(u.end for u in units)),
        pages=pages,
        heading_path=heading_path,
        prefix=prefix,
        counter=counter,
    )


def _chunk(
    *,
    index: int,
    text: str,
    span: Span,
    pages: list[int],
    heading_path: tuple[str, ...],
    prefix: str,
    counter: TokenCounter,
) -> Chunk:
    start, end = span
    body = text[start:end]
    embedding_text = prefix + body
    return Chunk(
        index=index,
        text=body,
        char_start=start,
        char_end=end,
        page_start=min(pages, default=None),
        page_end=max(pages, default=None),
        heading_path=heading_path,
        embedding_text=embedding_text,
        token_count=counter.count(embedding_text),
        content_hash=hashlib.sha256(embedding_text.encode("utf-8")).hexdigest(),
    )


# ------------------------------------------------------------------------------ fixed tokens


def _fixed_token_chunks(
    text: str, blocks: list[Block], config: ChunkingConfig, counter: TokenCounter
) -> list[Chunk]:
    starts = [b.start for b in blocks]

    def page_at(offset: int) -> int | None:
        i = bisect.bisect_right(starts, offset) - 1
        return blocks[i].page if i >= 0 else None

    chunks: list[Chunk] = []
    windows = _token_windows(text, 0, len(text), config.max_tokens, config.overlap_tokens, counter)
    for start, end in windows:
        pages = [p for p in (page_at(start), page_at(end - 1)) if p is not None]
        chunks.append(
            _chunk(
                index=len(chunks),
                text=text,
                span=(start, end),
                pages=pages,
                heading_path=(),
                prefix="",
                counter=counter,
            )
        )
    return chunks
