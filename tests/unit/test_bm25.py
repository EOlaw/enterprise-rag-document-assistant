from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from rag.sparse import Bm25Config, encode_document, encode_query, term_id, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Control AC-2(1) applies.", ["control", "ac-2(1)", "ac", "2", "1", "applies"]),
        ("RC.RP-01", ["rc.rp-01", "rc", "rp", "01"]),
        ("SP 800-53 Rev. 5", ["sp", "800-53", "800", "53", "rev", "5"]),
        ("ERR_CONN_RESET", ["err_conn_reset", "err", "conn", "reset"]),
        ("version v1.2.3.", ["version", "v1.2.3", "v1", "2", "3"]),
        ("The policy of the organization", ["policy", "organization"]),
        ("organization\u2019s plan", ["organization", "plan"]),  # possessive "s" dropped
        ("and/or", ["and/or"]),  # compound kept, stop-word parts dropped
    ],
)
def test_tokenizer_keeps_identifiers_whole_and_split(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_term_ids_are_stable_across_processes() -> None:
    probe = "from rag.sparse import term_id; print(term_id('rc.rp-01'))"
    other = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert int(other) == term_id("rc.rp-01")
    assert 0 <= term_id("x") < 2**32


def weights(text: str, config: Bm25Config) -> dict[int, float]:
    terms = encode_document(text, config)
    return dict(zip(terms.indices, terms.values, strict=True))


def test_document_weights_saturate_and_normalize_length() -> None:
    config = Bm25Config(avgdl=10)
    rid = term_id("ransomware")
    once = weights("ransomware alert", config)[rid]
    thrice = weights("ransomware ransomware ransomware alert", config)[rid]
    assert once < thrice < once * 3  # more occurrences help, sublinearly
    short = weights("ransomware", config)[rid]
    long = weights("ransomware " + "filler " * 30, config)[rid]
    assert short > long  # the same tf in a longer chunk weighs less
    flat = Bm25Config(avgdl=10, b=0)  # b=0 disables length normalization
    assert weights("ransomware", flat)[rid] == pytest.approx(
        weights("ransomware " + "filler " * 30, flat)[rid]
    )


def test_sparse_vectors_are_sorted_unique_and_deterministic() -> None:
    first = encode_document("incident incident response plan RC.RP-01", Bm25Config())
    assert list(first.indices) == sorted(set(first.indices))
    assert first == encode_document("incident incident response plan RC.RP-01", Bm25Config())
    query = encode_query("plan plan RC.RP-01")
    assert set(query.values) == {1.0} and len(query) == 5  # plan, rc.rp-01, rc, rp, 01


def test_empty_and_stop_word_only_text() -> None:
    assert len(encode_document("the of and", Bm25Config())) == 0
    assert len(encode_query("   ")) == 0


def test_config_is_validated_and_fingerprinted_into_metadata() -> None:
    with pytest.raises(ValidationError):
        Bm25Config(b=1.5)
    with pytest.raises(ValidationError):
        Bm25Config(avgdl=0)
    dumped = Bm25Config().model_dump()
    assert dumped == {"kind": "bm25", "tokenizer": "bm25-v1", "k1": 1.2, "b": 0.75, "avgdl": 74.0}
