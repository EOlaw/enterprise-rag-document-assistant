"""Qdrant index for one index version (one collection): dense and optional BM25 vectors.

Design rules:

* Payload holds **identifiers and filter fields only**. Chunk text stays in PostgreSQL and
  is returned only after the authorization/readiness re-check.
* Point IDs are deterministic (UUIDv5 of version, index fingerprint, chunk index), so a
  retried upsert overwrites instead of duplicating.
* Points are written with ``published=false`` and flipped to ``true`` only after every
  point is written and counted. PostgreSQL still decides visibility; the flag only keeps
  unfinished points out of the candidate list.
* Every write uses ``wait=True`` so a returned call means the operation is applied.
* Searches always require an organization and at least one knowledge base, and dense and
  sparse searches share one filter builder, so their scope cannot diverge.
* A sparse vector cannot be added to an existing collection (Qdrant returns 400), so
  BM25 requires a new index version and collection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.indexing.base import (
    ChunkPoint,
    VectorCollectionMismatchError,
    VectorHit,
    VectorStoreUnavailableError,
)
from rag.sparse.bm25 import SparseTerms

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
KEYWORD_FIELDS = ("kb_id", "document_id", "document_version_id", "document_index_id")
MAX_UPSERT_BATCH = 256


_TRANSPORT_ERRORS = (
    UnexpectedResponse,
    ResponseHandlingException,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


class QdrantVectorIndex:
    def __init__(
        self, client: QdrantClient, collection: str, dimension: int, *, sparse: bool = False
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.sparse = sparse

    # ------------------------------------------------------------------ collection

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes, or verify an existing one matches."""
        try:
            if not self.client.collection_exists(self.collection):
                try:
                    self.client.create_collection(
                        self.collection,
                        vectors_config={
                            DENSE_VECTOR: models.VectorParams(
                                size=self.dimension, distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config=(
                            {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                            if self.sparse
                            else None
                        ),
                    )
                except UnexpectedResponse as exc:
                    if exc.status_code != 409:  # another worker created it concurrently
                        raise
            self._verify_collection()
            self._ensure_payload_indexes()
        except _TRANSPORT_ERRORS as exc:
            raise VectorStoreUnavailableError("vector store request failed") from exc

    def _verify_collection(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        params = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if (
            params is None
            or params.size != self.dimension
            or params.distance != models.Distance.COSINE
        ):
            raise VectorCollectionMismatchError(
                f"collection {self.collection!r} does not have a {self.dimension}-d cosine "
                f"'{DENSE_VECTOR}' vector"
            )
        sparse_vectors = info.config.params.sparse_vectors or {}
        sparse = sparse_vectors.get(SPARSE_VECTOR)
        if self.sparse and (sparse is None or sparse.modifier != models.Modifier.IDF):
            raise VectorCollectionMismatchError(
                f"collection {self.collection!r} has no IDF-weighted '{SPARSE_VECTOR}' vector"
            )
        if not self.sparse and sparse is not None:
            raise VectorCollectionMismatchError(
                f"collection {self.collection!r} has a sparse vector its index version lacks"
            )

    def _ensure_payload_indexes(self) -> None:
        existing = set(self.client.get_collection(self.collection).payload_schema)
        wanted: dict[str, Any] = {
            # is_tenant co-locates each organization's vectors (Qdrant >= 1.11).
            "org_id": models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
            "published": models.PayloadSchemaType.BOOL,
            "chunk_index": models.PayloadSchemaType.INTEGER,
            **dict.fromkeys(KEYWORD_FIELDS, models.PayloadSchemaType.KEYWORD),
        }
        for field, schema in wanted.items():
            if field not in existing:
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema, wait=True
                )

    # ---------------------------------------------------------------------- writes

    def upsert(self, points: Sequence[ChunkPoint]) -> None:
        if len(points) > MAX_UPSERT_BATCH:
            raise ValueError(f"upsert at most {MAX_UPSERT_BATCH} points per call")
        for point in points:
            if len(point.vector) != self.dimension:
                raise VectorCollectionMismatchError("vector dimension does not match collection")
            if self.sparse != (point.sparse is not None):
                raise ValueError("sparse terms must be given exactly when the index is sparse")
        structs = [
            models.PointStruct(
                id=str(point.id),
                vector=self._vectors(point),
                payload={
                    "org_id": str(point.org_id),
                    "kb_id": str(point.kb_id),
                    "document_id": str(point.document_id),
                    "document_version_id": str(point.document_version_id),
                    "document_index_id": str(point.document_index_id),
                    "chunk_index": point.chunk_index,
                    "published": False,
                },
            )
            for point in points
        ]
        self._call(lambda: self.client.upsert(self.collection, points=structs, wait=True))

    @staticmethod
    def _vectors(point: ChunkPoint) -> dict[str, Any]:
        vectors: dict[str, Any] = {DENSE_VECTOR: list(point.vector)}
        if point.sparse is not None:
            vectors[SPARSE_VECTOR] = models.SparseVector(
                indices=list(point.sparse.indices), values=list(point.sparse.values)
            )
        return vectors

    def delete_extra_chunks(self, document_index_id: UUID, *, chunk_count: int) -> None:
        """Remove points left by an earlier attempt that produced more chunks."""
        self._call(
            lambda: self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            _match("document_index_id", document_index_id),
                            models.FieldCondition(
                                key="chunk_index", range=models.Range(gte=chunk_count)
                            ),
                        ]
                    )
                ),
                wait=True,
            )
        )

    def set_published(self, document_index_id: UUID, published: bool) -> None:
        self._call(
            lambda: self.client.set_payload(
                self.collection,
                payload={"published": published},
                points=models.Filter(must=[_match("document_index_id", document_index_id)]),
                wait=True,
            )
        )

    def delete_document_index(self, document_index_id: UUID) -> None:
        self._call(
            lambda: self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[_match("document_index_id", document_index_id)])
                ),
                wait=True,
            )
        )

    # ----------------------------------------------------------------------- reads

    def count(self, document_index_id: UUID, *, published: bool | None = None) -> int:
        conditions: list[models.Condition] = [_match("document_index_id", document_index_id)]
        if published is not None:
            conditions.append(
                models.FieldCondition(key="published", match=models.MatchValue(value=published))
            )
        result = self._call(
            lambda: self.client.count(
                self.collection, count_filter=models.Filter(must=conditions), exact=True
            )
        )
        return int(result.count)

    def search(
        self,
        vector: Sequence[float],
        *,
        org_id: UUID,
        kb_ids: Sequence[UUID],
        limit: int,
    ) -> list[VectorHit]:
        query_filter = self._scope(org_id, kb_ids, limit)
        if query_filter is None:
            return []
        return self._query(list(vector), DENSE_VECTOR, query_filter, limit)

    def search_sparse(
        self,
        terms: SparseTerms,
        *,
        org_id: UUID,
        kb_ids: Sequence[UUID],
        limit: int,
    ) -> list[VectorHit]:
        if not self.sparse:
            raise VectorCollectionMismatchError("this index version has no BM25 vectors")
        query_filter = self._scope(org_id, kb_ids, limit)
        if query_filter is None or not terms:
            return []  # no collection yet, or a query made only of stop words
        query = models.SparseVector(indices=list(terms.indices), values=list(terms.values))
        return self._query(query, SPARSE_VECTOR, query_filter, limit)

    def _scope(self, org_id: UUID, kb_ids: Sequence[UUID], limit: int) -> models.Filter | None:
        """The single tenant/KB/published filter used by every search."""
        if not kb_ids:
            raise ValueError("search requires at least one knowledge base")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not self._call(lambda: self.client.collection_exists(self.collection)):
            return None  # nothing has been published to this index version yet
        return models.Filter(
            must=[
                _match("org_id", org_id),
                models.FieldCondition(
                    key="kb_id", match=models.MatchAny(any=[str(kb) for kb in kb_ids])
                ),
                models.FieldCondition(key="published", match=models.MatchValue(value=True)),
            ]
        )

    def _query(
        self, query: Any, using: str, query_filter: models.Filter, limit: int
    ) -> list[VectorHit]:
        response = self._call(
            lambda: self.client.query_points(
                self.collection,
                query=query,
                using=using,
                query_filter=query_filter,
                limit=limit,
                with_payload=False,
                with_vectors=False,
            )
        )
        return [VectorHit(UUID(str(point.id)), float(point.score)) for point in response.points]

    def _call(self, operation: Any) -> Any:
        try:
            return operation()
        except _TRANSPORT_ERRORS as exc:
            raise VectorStoreUnavailableError("vector store request failed") from exc


def _match(key: str, value: UUID) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=str(value)))


def make_client(url: str, *, api_key: str | None = None, timeout: int = 30) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)
