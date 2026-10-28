"""Measurement-only heuristics used by evaluation reports (never by retrieval itself)."""

from __future__ import annotations

import re

# A dot leader: five or more periods, optionally space-separated (". . . . ." or "......").
_DOT_LEADER = re.compile(r"(?:\.\s?){5,}")
TOC_MIN_LEADERS = 2


def looks_like_table_of_contents(text: str) -> bool:
    """True for chunks dominated by table-of-contents lines.

    Heuristic: at least two dot-leader runs (``Section title . . . . . 26``). It detects the
    NIST tables of contents in the evaluation corpus; it will miss contents pages without
    dot leaders and can flag dotted tables. It is a diagnostic for reports, not a filter.
    """
    return len(_DOT_LEADER.findall(text)) >= TOC_MIN_LEADERS
