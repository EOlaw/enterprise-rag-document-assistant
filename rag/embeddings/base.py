"""Embedding contract and vector validation.

Every vector that enters the index or is used as a query passes :func:`validate_vectors`,
so a model swap, a mis-configured revision or a numerical failure cannot silently write
wrong-sized, non-finite or non-normalized vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag.errors import IngestionError

Vector = list[float]

NORM_TOLERANCE = 1e-3


class EmbeddingError(IngestionError):
    """Non-retryable embedding failure caused by inputs or configuration."""

    code = "embedding_error"


class EmbeddingInputTooLongError(EmbeddingError):
    code = "embedding_token_limit"


class EmbeddingModelMismatchError(EmbeddingError):
    code = "embedding_model_mismatch"


class InvalidEmbeddingError(EmbeddingError):
    code = "embedding_invalid_vector"


@runtime_checkable
class Embedder(Protocol):
    """A text embedder pinned to one model revision, producing unit-length vectors."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_seq_length(self) -> int:
        """Model sequence limit including special tokens (as recorded in index metadata)."""
        ...

    @property
    def max_input_tokens(self) -> int:
        """Content tokens accepted without truncation (excluding special tokens)."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def check_token_limits(embedder: Embedder, texts: Sequence[str]) -> list[int]:
    """Refuse inputs the model would silently truncate."""
    counts = embedder.count_tokens(texts)
    over = [i for i, count in enumerate(counts) if count > embedder.max_input_tokens]
    if over:
        raise EmbeddingInputTooLongError(
            f"{len(over)} input(s) exceed {embedder.max_input_tokens} tokens "
            f"(largest {max(counts[i] for i in over)}); they would be truncated"
        )
    return counts


def validate_vectors(vectors: Sequence[Sequence[float]], *, count: int, dimension: int) -> None:
    if len(vectors) != count:
        raise InvalidEmbeddingError(f"expected {count} vectors, got {len(vectors)}")
    for vector in vectors:
        if len(vector) != dimension:
            raise InvalidEmbeddingError(f"expected dimension {dimension}, got {len(vector)}")
        if not all(math.isfinite(value) for value in vector):
            raise InvalidEmbeddingError("vector contains non-finite values")
        norm = math.sqrt(sum(value * value for value in vector))
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise InvalidEmbeddingError(f"vector is not unit-normalized (norm={norm:.4f})")


def check_embedder_matches(
    embedder: Embedder, *, model_id: str, revision: str, dimension: int, max_seq_length: int
) -> None:
    """Ensure the loaded embedder is exactly the model recorded for an index version."""
    expected = (model_id, revision, dimension, max_seq_length)
    actual = (embedder.model_id, embedder.revision, embedder.dimension, embedder.max_seq_length)
    if actual != expected:
        raise EmbeddingModelMismatchError(
            f"embedder {actual} does not match index metadata {expected}"
        )
