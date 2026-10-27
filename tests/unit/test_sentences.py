from __future__ import annotations

import pytest

from rag.chunking.sentences import sentence_spans


def split(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


def test_basic_split() -> None:
    assert split("First one. Second one! Third? Yes.") == [
        "First one.",
        "Second one!",
        "Third?",
        "Yes.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "See Sec. 4.2 for details.",
        "Use MFA, e.g. TOTP or FIDO2 keys.",
        "Applies to the U.S. Army only.",
        "Approved by J. Smith yesterday.",
        "Version 2.0 is current.",
    ],
)
def test_abbreviations_and_decimals_do_not_split(text: str) -> None:
    assert split(text) == [text]


def test_closing_quote_stays_with_its_sentence() -> None:
    assert split('He wrote "Stop." Then left.') == ['He wrote "Stop."', "Then left."]


def test_newlines_are_hard_boundaries_and_whitespace_is_excluded() -> None:
    assert split("  - item one\n- item two  ") == ["- item one", "- item two"]


def test_spans_index_into_original_text() -> None:
    text = "Alpha beta.   Gamma delta."
    for start, end in sentence_spans(text):
        assert text[start:end] == text[start:end].strip()
