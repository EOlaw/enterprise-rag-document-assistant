"""Vector-store contract, value types and errors, free of heavy imports.

The worker's parser subprocess re-imports the worker module graph (``spawn`` start method).
Keeping ``qdrant_client`` (which loads numpy and grpc thread pools) out of this module keeps
that child small enough for its address-space limit. ``tests/unit/test_import_weight.py``
guards this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from rag.errors import IngestionError
from rag.sparse.bm25 import SparseTerms


class VectorStoreError(IngestionError):
    code = "vector_store_error"


class VectorStoreUnavailableError(VectorStoreError):
    code = "vector_store_unavailable"
    retryable = True


class VectorCollectionMismatchError(VectorStoreError):
    code = "vector_collection_mismatch"


class VectorCountMismatchError(VectorStoreError):
    code = "vector_count_mismatch"
    retryable = True


@dataclass(frozen=True, slots=True)
class ChunkPoint:
    id: UUID
    vector: Sequence[float]
    org_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_index_id: UUID
    chunk_index: int
    sparse: SparseTerms | None = None  # BM25 document weights; required by sparse indexes


@dataclass(frozen=True, slots=True)
class VectorHit:
    id: UUID
    # Cosine similarity (dense) or IDF-weighted BM25 (sparse). Comparable only within one
    # ranked list of one query; never a probability, never compared across retrievers.
    score: float


class VectorIndex(Protocol):
    """Operations the ingestion worker and search need from a vector store."""

    collection: str
    dimension: int
    sparse: bool  # whether the collection carries the BM25 sparse vector

    def ensure_collection(self) -> None: ...
    def upsert(self, points: Sequence[ChunkPoint]) -> None: ...
    def delete_extra_chunks(self, document_index_id: UUID, *, chunk_count: int) -> None: ...
    def set_published(self, document_index_id: UUID, published: bool) -> None: ...
    def delete_document_index(self, document_index_id: UUID) -> None: ...
    def count(self, document_index_id: UUID, *, published: bool | None = None) -> int: ...
    def search(
        self, vector: Sequence[float], *, org_id: UUID, kb_ids: Sequence[UUID], limit: int
    ) -> list[VectorHit]: ...
    def search_sparse(
        self, terms: SparseTerms, *, org_id: UUID, kb_ids: Sequence[UUID], limit: int
    ) -> list[VectorHit]: ...
