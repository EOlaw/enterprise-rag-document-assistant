"""Sentence-Transformers embedder pinned to an immutable Hugging Face commit.

The model is loaded lazily-imported so that parsing/chunking code paths never import torch.
On load we verify, rather than assume, the properties recorded in the index metadata:
embedding dimension, sequence limit and the presence of output normalization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag.embeddings.base import (
    EmbeddingModelMismatchError,
    Vector,
    check_token_limits,
    validate_vectors,
)
from rag.errors import ConfigurationError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
MAX_BATCH = 256


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        expected_dimension: int,
        expected_max_seq_length: int,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not _COMMIT.match(revision):
            raise ConfigurationError(
                "embedding revision must be a 40-character Hugging Face commit hash; "
                "branch names such as 'main' can change the model under an existing index"
            )
        if not 1 <= batch_size <= MAX_BATCH:
            raise ConfigurationError(f"batch_size must be between 1 and {MAX_BATCH}")
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 (torch is heavy)

        self._model: SentenceTransformer = SentenceTransformer(
            model_id, revision=revision, device=device
        )
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size
        # Renamed in sentence-transformers 6; keep compatibility with 5.x.
        get_dimension = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dimension = get_dimension()
        max_seq = self._model.max_seq_length
        if dimension != expected_dimension or max_seq != expected_max_seq_length:
            raise EmbeddingModelMismatchError(
                f"{model_id}@{revision} has dimension={dimension}, max_seq_length={max_seq}; "
                f"index expects {expected_dimension}, {expected_max_seq_length}"
            )
        if type(self._model[-1]).__name__ != "Normalize":
            # We normalize explicitly below either way; this documents the model contract.
            raise EmbeddingModelMismatchError(f"{model_id} does not end with a Normalize module")
        self._dimension = expected_dimension
        self._max_seq = expected_max_seq_length
        tokenizer = self._model.tokenizer
        self._special = len(tokenizer("", add_special_tokens=True)["input_ids"])

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
        return self._max_seq - self._special

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        encoded = self._model.tokenizer(
            list(texts), add_special_tokens=False, truncation=False, verbose=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        check_token_limits(self, texts)
        array = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors: list[Vector] = array.astype("float64").tolist()
        validate_vectors(vectors, count=len(texts), dimension=self._dimension)
        return vectors
