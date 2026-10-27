"""Deterministic test doubles. The real MiniLM model is exercised in model-marked tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from rag.chunking.tokens import RegexTokenCounter
from rag.embeddings import Vector, check_token_limits, validate_vectors


class HashingEmbedder:
    """Signed feature-hashing bag of words, unit-normalized.

    Lexically similar texts get similar vectors, which makes search assertions meaningful
    without loading torch. It matches the test index metadata (regex-v1 @ test-v1).
    """

    def __init__(
        self,
        *,
        model_id: str = "regex-v1",
        revision: str = "test-v1",
        dimension: int = 384,
        max_seq_length: int = 256,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._dimension = dimension
        self._max_seq = max_seq_length
        self._counter = RegexTokenCounter()
        self.embedded_texts = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return self._max_seq

    @property
    def max_input_tokens(self) -> int:
        return self._max_seq - 2

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [self._counter.count(text) for text in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        check_token_limits(self, texts)
        vectors = [self._vector(text) for text in texts]
        validate_vectors(vectors, count=len(texts), dimension=self._dimension)
        self.embedded_texts += len(texts)
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]

    def _vector(self, text: str) -> Vector:
        values = [0.0] * self._dimension
        lowered = text.lower()
        for start, end in self._counter.spans(lowered):
            digest = hashlib.blake2b(lowered[start:end].encode(), digest_size=16).digest()
            # Three signed buckets per token so one hash collision cannot cancel a term.
            for offset in (0, 5, 10):
                bucket = int.from_bytes(digest[offset : offset + 4], "little") % self._dimension
                values[bucket] += 1.0 if digest[offset + 4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            values[0], norm = 1.0, 1.0
        return [v / norm for v in values]
