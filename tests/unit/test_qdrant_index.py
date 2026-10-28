"""QdrantVectorIndex behaviour in qdrant-client local mode (server behaviour is covered by
the integration suite when TEST_QDRANT_URL is set)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient, models

from rag.indexing import (
    ChunkPoint,
    VectorCollectionMismatchError,
    VectorStoreUnavailableError,
)
from rag.indexing.qdrant import QdrantVectorIndex
from rag.sparse import Bm25Config, encode_document, encode_query
from tests.fakes import HashingEmbedder

ORG, OTHER_ORG = uuid4(), uuid4()
KB, OTHER_KB = uuid4(), uuid4()
EMBEDDER = HashingEmbedder(dimension=64)


@pytest.fixture
def index() -> Iterator[QdrantVectorIndex]:
    client = QdrantClient(location=":memory:")
    vectors = QdrantVectorIndex(client, "chunks", 64)
    vectors.ensure_collection()
    yield vectors
    client.close()


def points(
    document_index_id: UUID, texts: list[str], *, org: UUID = ORG, kb: UUID = KB
) -> list[ChunkPoint]:
    return [
        ChunkPoint(
            id=uuid4(),
            vector=EMBEDDER.embed_query(text),
            org_id=org,
            kb_id=kb,
            document_id=document_index_id,
            document_version_id=document_index_id,
            document_index_id=document_index_id,
            chunk_index=i,
        )
        for i, text in enumerate(texts)
    ]


def test_unpublished_points_are_not_searchable(index: QdrantVectorIndex) -> None:
    doc = uuid4()
    index.upsert(points(doc, ["incident response", "budget planning"]))
    query = EMBEDDER.embed_query("incident response")
    assert index.search(query, org_id=ORG, kb_ids=[KB], limit=5) == []
    assert index.count(doc) == 2 and index.count(doc, published=True) == 0
    index.set_published(doc, True)
    hits = index.search(query, org_id=ORG, kb_ids=[KB], limit=5)
    assert len(hits) == 2 and hits[0].score >= hits[1].score


def test_search_filters_by_organization_and_knowledge_base(index: QdrantVectorIndex) -> None:
    mine, theirs, other_kb = uuid4(), uuid4(), uuid4()
    index.upsert(points(mine, ["incident response"]))
    index.upsert(points(theirs, ["incident response"], org=OTHER_ORG, kb=OTHER_KB))
    index.upsert(points(other_kb, ["incident response"], kb=OTHER_KB))
    for doc in (mine, theirs, other_kb):
        index.set_published(doc, True)
    query = EMBEDDER.embed_query("incident response")
    assert len(index.search(query, org_id=ORG, kb_ids=[KB], limit=10)) == 1
    assert len(index.search(query, org_id=ORG, kb_ids=[KB, OTHER_KB], limit=10)) == 2
    assert len(index.search(query, org_id=OTHER_ORG, kb_ids=[KB], limit=10)) == 0
    with pytest.raises(ValueError):
        index.search(query, org_id=ORG, kb_ids=[], limit=10)


def test_deterministic_ids_overwrite_and_extra_chunks_are_removed(index: QdrantVectorIndex) -> None:
    doc = uuid4()
    first = points(doc, ["a", "b", "c"])
    index.upsert(first)
    index.upsert(first[:2])  # retry with identical ids
    assert index.count(doc) == 3
    index.delete_extra_chunks(doc, chunk_count=2)  # a later attempt produced 2 chunks
    assert index.count(doc) == 2
    index.delete_document_index(doc)
    assert index.count(doc) == 0


def test_ensure_collection_is_idempotent_and_detects_mismatch() -> None:
    client = QdrantClient(location=":memory:")
    QdrantVectorIndex(client, "chunks", 64).ensure_collection()
    QdrantVectorIndex(client, "chunks", 64).ensure_collection()
    with pytest.raises(VectorCollectionMismatchError):
        QdrantVectorIndex(client, "chunks", 32).ensure_collection()
    client.create_collection(
        "unnamed", vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE)
    )
    with pytest.raises(VectorCollectionMismatchError):
        QdrantVectorIndex(client, "unnamed", 64).ensure_collection()


def test_wrong_dimension_points_are_refused(index: QdrantVectorIndex) -> None:
    good = points(uuid4(), ["x"])[0]
    bad = ChunkPoint(good.id, [1.0], good.org_id, good.kb_id, good.document_id,
                     good.document_version_id, good.document_index_id, 0)  # fmt: skip
    with pytest.raises(VectorCollectionMismatchError):
        index.upsert([bad])


def test_missing_collection_searches_empty() -> None:
    vectors = QdrantVectorIndex(QdrantClient(location=":memory:"), "absent", 64)
    assert vectors.search(EMBEDDER.embed_query("x"), org_id=ORG, kb_ids=[KB], limit=3) == []


def test_unreachable_server_is_a_retryable_error() -> None:
    client = QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    with pytest.raises(VectorStoreUnavailableError) as error:
        QdrantVectorIndex(client, "chunks", 64).ensure_collection()
    assert error.value.retryable and error.value.code == "vector_store_unavailable"


# ------------------------------------------------------------------- BM25 sparse


def sparse_points(
    document_index_id: UUID, texts: list[str], *, org: UUID = ORG
) -> list[ChunkPoint]:
    return [
        replace(point, sparse=encode_document(text, Bm25Config()))
        for point, text in zip(points(document_index_id, texts, org=org), texts, strict=True)
    ]


@pytest.fixture
def hybrid() -> Iterator[QdrantVectorIndex]:
    client = QdrantClient(location=":memory:")
    vectors = QdrantVectorIndex(client, "hybrid", 64, sparse=True)
    vectors.ensure_collection()
    yield vectors
    client.close()


def test_sparse_search_finds_identifiers_within_scope(hybrid: QdrantVectorIndex) -> None:
    mine, theirs = uuid4(), uuid4()
    hybrid.upsert(sparse_points(mine, ["Control RC.RP-01 recovery", "budget forecast"]))
    hybrid.upsert(sparse_points(theirs, ["Control RC.RP-01 recovery"], org=OTHER_ORG))
    for doc in (mine, theirs):
        hybrid.set_published(doc, True)
    hits = hybrid.search_sparse(encode_query("RC.RP-01"), org_id=ORG, kb_ids=[KB], limit=10)
    assert len(hits) == 1 and hits[0].score > 0
    assert hybrid.search_sparse(encode_query("the of"), org_id=ORG, kb_ids=[KB], limit=5) == []


def test_sparse_terms_must_match_the_index_kind(
    hybrid: QdrantVectorIndex, index: QdrantVectorIndex
) -> None:
    with pytest.raises(ValueError, match="sparse terms"):
        hybrid.upsert(points(uuid4(), ["no sparse terms"]))
    with pytest.raises(ValueError, match="sparse terms"):
        index.upsert(sparse_points(uuid4(), ["unexpected sparse terms"]))
    with pytest.raises(VectorCollectionMismatchError):
        index.search_sparse(encode_query("x"), org_id=ORG, kb_ids=[KB], limit=5)


def test_sparse_and_dense_collections_are_not_interchangeable() -> None:
    client = QdrantClient(location=":memory:")
    QdrantVectorIndex(client, "dense_only", 64).ensure_collection()
    with pytest.raises(VectorCollectionMismatchError, match="IDF"):
        QdrantVectorIndex(client, "dense_only", 64, sparse=True).ensure_collection()
    QdrantVectorIndex(client, "both", 64, sparse=True).ensure_collection()
    with pytest.raises(VectorCollectionMismatchError, match="sparse vector"):
        QdrantVectorIndex(client, "both", 64).ensure_collection()


def test_delete_and_count_tolerate_a_missing_collection() -> None:
    vectors = QdrantVectorIndex(QdrantClient(location=":memory:"), "never_created", 64)
    vectors.delete_document_index(uuid4())
    assert vectors.count(uuid4()) == 0
