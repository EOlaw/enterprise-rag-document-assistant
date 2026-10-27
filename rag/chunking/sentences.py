"""Conservative rule-based sentence segmentation returning character spans.

Under-splitting is preferred over over-splitting: a missed boundary only makes a unit
larger (it is then split by tokens), while a false boundary inside "Sec. 4.2" or "e.g."
separates text that belongs together.
"""

from __future__ import annotations

import re

from rag.chunking.tokens import Span

# Group 1 captures closing quotes/brackets, which belong to the sentence they close.
_BOUNDARY = re.compile(r"(?<=[.!?])([\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "vs",
        "cf",
        "al",
        "fig",
        "figs",
        "sec",
        "secs",
        "no",
        "nos",
        "vol",
        "pp",
        "p",
        "ch",
        "art",
        "para",
        "approx",
        "dept",
        "inc",
        "ltd",
        "corp",
        "co",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "jr",
        "sr",
        "st",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "u.s",
        "u.k",
        "rev",
        "ref",
        "eq",
    }
)
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.\W*$")


def sentence_spans(text: str) -> list[Span]:
    """Split ``text`` into sentence spans (whitespace between sentences is excluded)."""
    spans: list[Span] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start() + len(match.group(1) or "")
        candidate = text[start:end]
        if match.group(1) is not None and _ends_with_abbreviation(candidate):
            continue
        _append(spans, text, start, end)
        start = match.end()
    _append(spans, text, start, len(text))
    return spans


def _ends_with_abbreviation(candidate: str) -> bool:
    match = _LAST_WORD.search(candidate.rstrip("\"')]"))
    if not match:
        return False
    word = match.group(1).lower()
    return word in _ABBREVIATIONS or len(word) == 1  # initials such as "J. Smith"


def _append(spans: list[Span], text: str, start: int, end: int) -> None:
    segment = text[start:end]
    stripped_start = len(segment) - len(segment.lstrip())
    stripped_end = len(segment.rstrip())
    if stripped_end > stripped_start:
        spans.append((start + stripped_start, start + stripped_end))
