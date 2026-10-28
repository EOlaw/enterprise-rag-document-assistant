"""Cross-encoder reranking with a pinned model and explicit pair-length limits.

The model scores (query, passage) pairs jointly. Its outputs here are raw logits
(``ms-marco-MiniLM-L6-v2`` uses an identity activation): useful for ordering candidates
of one query, **not** probabilities, not calibrated, and not evidence of correctness,
recency or authority. Reranking reorders what retrieval found. It cannot know that a
withdrawn publication should lose to its replacement.

Pairs longer than the model window are refused, not truncated. With chunks of at most
254 embedding tokens and queries of at most 254, pairs fit in 512 positions; the check
makes that invariant explicit if either limit changes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rag.errors import ConfigurationError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
MAX_RERANK_CANDIDATES = 100


class RerankInputTooLongError(ValueError):
    pass


class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def max_length(self) -> int: ...

    def pair_tokens(self, query: str, passages: Sequence[str]) -> list[int]: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        expected_max_length: int = 512,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        if not _COMMIT.match(revision):
            raise ConfigurationError("reranker revision must be a 40-character commit hash")
        from sentence_transformers import CrossEncoder  # noqa: PLC0415 (torch is heavy)

        self._model: CrossEncoder = CrossEncoder(model_id, revision=revision, device=device)
        window = self._window()
        if window != expected_max_length or self._model.num_labels != 1:
            raise ConfigurationError(
                f"{model_id}@{revision} has max_length={window}, "
                f"num_labels={self._model.num_labels}; expected {expected_max_length}, 1"
            )
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def max_length(self) -> int:
        return self._window()

    def _window(self) -> int:
        # Renamed to max_seq_length in sentence-transformers 6; max_length is deprecated.
        value = getattr(self._model, "max_seq_length", None)
        return int(value if value is not None else self._model.max_length)

    def pair_tokens(self, query: str, passages: Sequence[str]) -> list[int]:
        tokenizer = self._model.tokenizer
        return [
            len(tokenizer(query, passage, add_special_tokens=True, truncation=False)["input_ids"])
            for passage in passages
        ]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        lengths = self.pair_tokens(query, passages)
        if max(lengths) > self.max_length:
            raise RerankInputTooLongError(
                f"a query/passage pair has {max(lengths)} tokens; the reranker accepts "
                f"{self.max_length} and would silently truncate"
            )
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in scores]


@dataclass(frozen=True, slots=True)
class Reranked:
    index: int  # position in the input list
    score: float


def rerank_order(scores: Sequence[float]) -> list[Reranked]:
    """Order by score descending; ties keep the incoming (retrieval) order."""
    return sorted((Reranked(i, s) for i, s in enumerate(scores)), key=lambda r: (-r.score, r.index))
