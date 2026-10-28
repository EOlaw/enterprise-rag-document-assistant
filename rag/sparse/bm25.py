"""BM25 term weighting for Qdrant sparse vectors (pure Python, no heavy imports).

Split of responsibilities:

* documents carry BM25's term-frequency saturation
  ``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``;
* queries carry weight 1 per distinct term;
* Qdrant's ``idf`` modifier multiplies each matched term by its inverse document
  frequency at query time, so IDF stays current as documents are added.

Verified on Qdrant 1.18.3: that IDF is computed over the **whole collection**, not over
the filtered subset. Every tenant and every not-yet-cleaned-up point in an index
version's collection influences BM25 scores (statistics only, never text).

``avgdl`` is a fixed configuration value because points already written cannot be
re-weighted when the corpus grows. It is part of the index fingerprint.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TOKENIZER_VERSION: Literal["bm25-v1"] = "bm25-v1"

# Words and identifier-like compounds: "ac-2(1)", "rc.rp-01", "800-53", "err_conn_reset",
# "v1.2.3", "doi.org/10.6028". A connector must sit between two alphanumeric runs, so a
# sentence-final period is not part of a token.
_TOKEN = re.compile(r"[^\W_]+(?:[-._/:][^\W_]+)*(?:\(\d+\))?")
_PART = re.compile(r"[^\W_]+")
_STOP_WORDS = frozenset(
    """a an and are as at be been but by for from has have if in into is it its of on or such
    that the their then there these this those to was were will with which who whom what
    when where why how can could should would may might must shall do does did not no than
    so also any all each other some our we you they he she them his her""".split()  # noqa: SIM905
)


class Bm25Config(BaseModel):
    """BM25 parameters recorded in ``index_versions.sparse_config``.

    k1 and b are the textbook defaults. avgdl is the mean BM25 token count of
    default-configured chunks measured on this repository's own documentation, a neutral
    corpus, not the evaluation corpus (see docs/architecture/hybrid-retrieval.md).
    None of these values were tuned against evaluation questions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["bm25"] = "bm25"
    tokenizer: Literal["bm25-v1"] = TOKENIZER_VERSION
    k1: float = Field(default=1.2, gt=0, le=3)
    b: float = Field(default=0.75, ge=0, le=1)
    # Measured 2026-10-27: 117 default chunks from the 8 docs/ Markdown files, mean 73.9.
    avgdl: float = Field(default=74.0, gt=0)


@dataclass(frozen=True, slots=True)
class SparseTerms:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.indices)


def tokenize(text: str) -> list[str]:
    """Lowercased terms; identifier compounds are emitted whole **and** as their parts."""
    terms: list[str] = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group()
        parts = _PART.findall(token)
        if len(parts) > 1:
            terms.append(token)
            terms.extend(part for part in parts if part not in _STOP_WORDS)
        elif token not in _STOP_WORDS and not (len(token) == 1 and token.isalpha()):
            terms.append(token)
    return terms


def term_id(term: str) -> int:
    """Stable 32-bit term index (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "little")


def _to_sparse(weights: dict[int, float]) -> SparseTerms:
    ordered = sorted(weights.items())
    return SparseTerms(tuple(i for i, _ in ordered), tuple(w for _, w in ordered))


def encode_document(text: str, config: Bm25Config) -> SparseTerms:
    terms = tokenize(text)
    if not terms:
        return SparseTerms((), ())
    norm = config.k1 * (1 - config.b + config.b * len(terms) / config.avgdl)
    weights: dict[int, float] = {}
    for term, frequency in Counter(terms).items():
        index = term_id(term)  # colliding terms share a slot; their weights add up
        weights[index] = weights.get(index, 0.0) + (
            frequency * (config.k1 + 1) / (frequency + norm)
        )
    return _to_sparse(weights)


def encode_query(text: str) -> SparseTerms:
    weights: dict[int, float] = {}
    for term in set(tokenize(text)):
        index = term_id(term)
        weights[index] = weights.get(index, 0.0) + 1.0
    return _to_sparse(weights)
