"""Span-based relevance: gold labels are quotes, not chunk IDs.

A label ``(document, quote)`` is located in that document's canonical text. A retrieved
chunk *covers* a span when it contains at least ``COVER_FRACTION`` of it, and *partially*
overlaps when it contains any of it. This keeps one label set valid across different
chunking strategies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

COVER_FRACTION = 0.5
_WS = re.compile(r"\s+")


class QuoteNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class Span:
    document: str  # dataset document key
    start: int
    end: int
    grade: int  # 2 = directly answers; 1 = supporting context


@dataclass(frozen=True, slots=True)
class RetrievedSpan:
    document: str
    start: int
    end: int


def locate_quote(quote: str, canonical_text: str) -> tuple[int, int]:
    """Find ``quote`` in ``canonical_text`` ignoring whitespace differences.

    Raises :class:`QuoteNotFoundError` when absent or ambiguous (found more than once), so
    a label can never silently attach to the wrong passage.
    """
    matches = locate_quote_all(quote, canonical_text)
    if len(matches) > 1:
        raise QuoteNotFoundError(f"quote is ambiguous ({len(matches)} matches): {quote[:60]!r}")
    return matches[0]


def locate_quote_all(quote: str, canonical_text: str) -> list[tuple[int, int]]:
    """Every occurrence, for labels that deliberately accept any repeated passage."""
    words = quote.split()
    if not words:
        raise QuoteNotFoundError("empty quote")
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    matches = [m.span() for m in pattern.finditer(canonical_text)]
    if not matches:
        raise QuoteNotFoundError(f"quote not found: {quote[:60]!r}")
    return matches


def overlap_fraction(span: Span, retrieved: RetrievedSpan) -> float:
    if span.document != retrieved.document:
        return 0.0
    overlap = min(span.end, retrieved.end) - max(span.start, retrieved.start)
    return max(0, overlap) / (span.end - span.start)


def result_gains(spans: Sequence[Span], ranked: Sequence[RetrievedSpan]) -> list[int]:
    """Graded gain per rank; each span contributes at most its grade in total.

    Covering a span is worth its grade; a partial overlap of a grade-2 span is worth 1.
    A result earns the largest *remaining* credit of any span it touches, so overlapping
    chunks that repeat the same passage are not rewarded twice and nDCG stays <= 1.
    """
    given = [0] * len(spans)
    gains: list[int] = []
    for result in ranked:
        best, best_index = 0, -1
        for i, span in enumerate(spans):
            fraction = overlap_fraction(span, result)
            if fraction >= COVER_FRACTION:
                target = span.grade
            elif fraction > 0 and span.grade == 2:
                target = 1
            else:
                target = 0
            if target - given[i] > best:
                best, best_index = target - given[i], i
        if best_index >= 0:
            given[best_index] += best
        gains.append(best)
    return gains


def covered_spans(spans: Sequence[Span], ranked: Sequence[RetrievedSpan]) -> set[int]:
    return {
        i
        for i, span in enumerate(spans)
        if any(overlap_fraction(span, result) >= COVER_FRACTION for result in ranked)
    }


def first_covering_rank(spans: Sequence[Span], ranked: Sequence[RetrievedSpan]) -> int | None:
    for rank, result in enumerate(ranked, start=1):
        if any(overlap_fraction(span, result) >= COVER_FRACTION for span in spans):
            return rank
    return None
