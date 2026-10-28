"""Embedding, vector indexing, publication gating and scoped dense search.

Runs against SQLite and (with TEST_DATABASE_URL) PostgreSQL, and against qdrant-client's
local mode or (with TEST_QDRANT_URL) a real Qdrant server.
"""

from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

import pytest
from kip.cli import activate_index, create_index, requeue_processed
from kip.db.models import (
    Document,
    DocumentChunk,
    DocumentIndex,
    DocumentVersion,
    EmbeddingModel,
    IndexVersion,
    IngestionJob,
    KnowledgeBase,
    utcnow,
)
from kip.retrieval import SearchScopeError, dense_search
from qdrant_client import models
from sqlalchemy import func, select

from rag.config import ChunkingConfig
from rag.indexing import ChunkPoint, VectorStoreUnavailableError
from rag.indexing.qdrant import QdrantVectorIndex
from rag.retrieval import QueryTooLongError
from tests.fakes import HashingEmbedder
from tests.integration.conftest import Harness, InlineProcessor

RANSOMWARE = (
    b"Ransomware incidents must be escalated to the security operations lead within one hour."
)
BUDGET = b"The facilities team prepares the quarterly budget forecast every March."


class FlakyIndex:
    """Delegates to a real index but fails selected operations a set number of times."""

    def __init__(self, inner: QdrantVectorIndex, fail: dict[str, int]) -> None:
        self.inner = inner
        self.fail = fail  # shared by every index this factory creates
        self.collection = inner.collection
        self.dimension = inner.dimension
        self.sparse = inner.sparse

    def _maybe_fail(self, operation: str) -> None:
        if self.fail.get(operation, 0) > 0:
            self.fail[operation] -= 1
            raise VectorStoreUnavailableError(f"injected {operation} failure")

    def ensure_collection(self) -> None:
        self._maybe_fail("ensure_collection")
        self.inner.ensure_collection()

    def upsert(self, points: Sequence[ChunkPoint]) -> None:
        self._maybe_fail("upsert")
        self.inner.upsert(points)

    def delete_extra_chunks(self, document_index_id: UUID, *, chunk_count: int) -> None:
        self.inner.delete_extra_chunks(document_index_id, chunk_count=chunk_count)

    def set_published(self, document_index_id: UUID, published: bool) -> None:
        self.inner.set_published(document_index_id, published)

    def delete_document_index(self, document_index_id: UUID) -> None:
        self._maybe_fail("delete")
        self.inner.delete_document_index(document_index_id)

    def count(self, document_index_id: UUID, *, published: bool | None = None) -> int:
        if published:
            self._maybe_fail("count_published")
        return self.inner.count(document_index_id, published=published)

    def search(self, vector, *, org_id, kb_ids, limit):  # type: ignore[no-untyped-def]
        return self.inner.search(vector, org_id=org_id, kb_ids=kb_ids, limit=limit)

    def search_sparse(self, terms, *, org_id, kb_ids, limit):  # type: ignore[no-untyped-def]
        return self.inner.search_sparse(terms, org_id=org_id, kb_ids=kb_ids, limit=limit)


def flaky(harness: Harness, **fail: int):  # type: ignore[no-untyped-def]
    remaining = dict(fail)

    def factory(index: IndexVersion, model: EmbeddingModel) -> FlakyIndex:
        inner = harness.vectors(index, model)
        assert isinstance(inner, QdrantVectorIndex)
        return FlakyIndex(inner, remaining)

    return factory


def make_due(harness: Harness, job_id: UUID) -> None:
    with harness.sessions.begin() as session:
        session.get(IngestionJob, job_id).run_after = utcnow() - timedelta(seconds=1)


def point_payloads(harness: Harness, index_id: UUID | None = None) -> list[dict]:
    points, _ = harness.client.scroll(harness.collection(index_id), limit=1000, with_payload=True)
    return [point.payload for point in points]


# ------------------------------------------------------------------- publication


def test_publish_and_search_returns_exact_citation_text(harness: Harness) -> None:
    submission = harness.submit(RANSOMWARE)
    harness.submit(BUDGET, filename="budget.txt")
    worker = harness.worker()
    assert worker.run_once() and worker.run_once()

    result = harness.search("who handles ransomware escalation")
    assert result.results[0].document_id == submission.document_id
    top = result.results[0]
    with harness.sessions() as session:
        version = session.get(DocumentVersion, submission.version_id)
        canonical = harness.store.get(version.canonical_text_key, max_bytes=10_000).decode()
    assert canonical[top.char_start : top.char_end] == top.text
    assert top.version_number == 1 and top.filename == "guide.txt"
    assert result.dropped_by_recheck == 0
    payloads = point_payloads(harness)
    assert all(p["published"] is True for p in payloads)
    assert all("text" not in p for p in payloads)  # text lives only in PostgreSQL


def test_parsed_title_replaces_filename_default_only(harness: Harness) -> None:
    markdown = b"---\ntitle: Incident Runbook\n---\n# Runbook\n\nIsolate the host first."
    adopted = harness.submit(markdown, filename="runbook.md")
    explicit = harness.submit(b"# Other\n\nSome text.", filename="other.md")
    with harness.sessions.begin() as session:
        session.get(Document, explicit.document_id).title = "Chosen by an operator"
    worker = harness.worker()
    assert worker.run_once() and worker.run_once()
    with harness.sessions() as session:
        assert session.get(Document, adopted.document_id).title == "Incident Runbook"
        assert session.get(Document, explicit.document_id).title == "Chosen by an operator"
    assert harness.search("isolate the host").results[0].title == "Incident Runbook"


def test_retry_after_success_does_not_duplicate_vectors(harness: Harness) -> None:
    submission = harness.submit(("Recovery evidence sentence. " * 200).encode())
    worker = harness.worker()
    assert worker.run_once()
    first = {p.id for p in harness.client.scroll(harness.collection(), limit=1000)[0]}
    with harness.sessions.begin() as session:
        index = session.scalar(select(DocumentIndex))
        index.status = "building"  # simulate a crash right before the publication commit
        job = session.get(IngestionJob, submission.job_id)
        job.status, job.stage = "queued", "queued"
    assert worker.run_once()
    second = {p.id for p in harness.client.scroll(harness.collection(), limit=1000)[0]}
    assert first == second and len(first) > 1
    with harness.sessions() as session:
        assert session.scalar(select(DocumentIndex)).status == "ready"


def test_vector_store_outage_retries_and_stays_invisible(harness: Harness) -> None:
    submission = harness.submit(RANSOMWARE)
    assert harness.worker(vector_index_factory=flaky(harness, upsert=1)).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.error_code) == ("retrying", "vector_store_unavailable")
        assert session.get(DocumentVersion, submission.version_id).status == "processed"
        assert session.get(Document, submission.document_id).current_version_id is None
        assert session.scalar(select(DocumentIndex)).status == "building"
    assert harness.search("ransomware").results == []

    make_due(harness, submission.job_id)

    class MustNotReparse(InlineProcessor):
        def process(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("processed chunks should be reused")

    assert harness.worker(MustNotReparse()).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.stage, job.attempt) == ("completed", "published", 2)
    assert harness.search("ransomware").results


def test_published_vectors_without_postgres_commit_are_never_returned(harness: Harness) -> None:
    # Qdrant flips published=true, then the verification step fails before PostgreSQL
    # commits readiness. The points are visible to Qdrant filters but not to search.
    submission = harness.submit(RANSOMWARE)
    assert harness.worker(vector_index_factory=flaky(harness, count_published=1)).run_once()
    assert {p["published"] for p in point_payloads(harness)} == {True}
    result = harness.search("ransomware escalation")
    assert result.results == [] and result.dropped_by_recheck == result.candidates > 0
    make_due(harness, submission.job_id)
    assert harness.worker().run_once()
    assert harness.search("ransomware escalation").results


def test_unrecoverable_vector_failures_exhaust_attempts(harness: Harness) -> None:
    submission = harness.submit(RANSOMWARE)
    for _ in range(3):
        assert harness.worker(vector_index_factory=flaky(harness, ensure_collection=9)).run_once()
        make_due(harness, submission.job_id)
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.attempt) == ("failed", 3)
        assert session.get(Document, submission.document_id).current_version_id is None


# -------------------------------------------------- configuration and model guards


def test_embedder_that_differs_from_index_metadata_is_rejected(harness: Harness) -> None:
    submission = harness.submit(RANSOMWARE)
    wrong = HashingEmbedder(dimension=128)
    assert harness.worker(embedder=wrong).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.error_code) == ("failed", "embedding_model_mismatch")
    assert harness.client.collection_exists(harness.collection()) is False


def test_existing_collection_with_wrong_dimension_is_not_written(harness: Harness) -> None:
    harness.client.create_collection(
        harness.collection(),
        vectors_config={"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)},
    )
    submission = harness.submit(RANSOMWARE)
    assert harness.worker().run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.error_code) == ("failed", "vector_collection_mismatch")
    assert harness.client.count(harness.collection()).count == 0


def test_chunks_longer_than_the_model_window_fail_instead_of_truncating(harness: Harness) -> None:
    submission = harness.submit(("word " * 150).encode())
    tight = HashingEmbedder()
    tight._max_seq = 64  # the index says 256, so this is also a metadata mismatch
    assert harness.worker(embedder=tight).run_once()
    with harness.sessions() as session:
        assert session.get(IngestionJob, submission.job_id).error_code == "embedding_model_mismatch"


def test_query_longer_than_model_window_is_refused(harness: Harness) -> None:
    with pytest.raises(QueryTooLongError):
        harness.search("word " * 300)


# ------------------------------------------------------------ versions and deletion


def test_new_version_supersedes_old_and_old_vectors_are_unpublished(harness: Harness) -> None:
    first = harness.submit(b"Retention period for incident records is three years.")
    worker = harness.worker()
    assert worker.run_once()
    second = harness.submit(
        b"Retention period for incident records is seven years.", document_id=first.document_id
    )
    assert worker.run_once()
    result = harness.search("incident record retention period")
    assert [r.document_version_id for r in result.results] == [second.version_id]
    assert "seven years" in result.results[0].text
    with harness.sessions() as session:
        assert session.get(DocumentVersion, first.version_id).status == "superseded"
        assert session.get(DocumentVersion, second.version_id).status == "ready"
    published = {p["document_version_id"]: p["published"] for p in point_payloads(harness)}
    assert published == {str(first.version_id): False, str(second.version_id): True}


def test_older_version_finishing_late_never_moves_document_backwards(harness: Harness) -> None:
    first = harness.submit(b"Old guidance: rotate passwords every 30 days.")
    second = harness.submit(
        b"New guidance: do not force periodic password rotation.", document_id=first.document_id
    )
    with harness.sessions.begin() as session:  # make version 2 run first
        session.get(IngestionJob, first.job_id).run_after = utcnow() + timedelta(minutes=5)
    worker = harness.worker()
    assert worker.run_once()
    make_due(harness, first.job_id)
    assert worker.run_once()
    with harness.sessions() as session:
        assert session.get(Document, first.document_id).current_version_id == second.version_id
        assert session.get(DocumentVersion, first.version_id).status == "superseded"
        assert session.scalar(select(func.count()).select_from(DocumentIndex)) == 2
    texts = [r.text for r in harness.search("password rotation guidance").results]
    assert texts == ["New guidance: do not force periodic password rotation."]


def test_document_deleted_during_indexing_is_not_published_and_vectors_removed(
    harness: Harness,
) -> None:
    submission = harness.submit(RANSOMWARE)

    class DeleteDuringEmbedding(HashingEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            with harness.sessions.begin() as session:
                session.get(Document, submission.document_id).deleted_at = utcnow()
            return super().embed_documents(texts)

    assert harness.worker(embedder=DeleteDuringEmbedding()).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.error_code) == ("failed", "document_deleted")
        assert session.scalar(select(DocumentIndex)).status == "building"
    assert harness.client.count(harness.collection()).count == 0


def test_deletion_after_publication_hides_results_immediately(harness: Harness) -> None:
    submission = harness.submit(RANSOMWARE)
    assert harness.worker().run_once()
    assert harness.search("ransomware").results
    with harness.sessions.begin() as session:
        session.get(Document, submission.document_id).deleted_at = utcnow()
    result = harness.search("ransomware")
    # Qdrant still holds published points; PostgreSQL decides they are not visible.
    assert result.results == [] and result.candidates > 0


def test_deleted_knowledge_base_is_rejected_as_scope(harness: Harness) -> None:
    harness.submit(RANSOMWARE)
    assert harness.worker().run_once()
    with harness.sessions.begin() as session:
        session.get(KnowledgeBase, harness.kb.id).deleted_at = utcnow()
    with pytest.raises(SearchScopeError):
        harness.search("ransomware")


# ------------------------------------------------------------- tenant isolation


def test_cross_tenant_isolation_with_identical_content(harness: Harness) -> None:
    other_org, other_kb = harness.add_tenant("other")
    mine = harness.submit(RANSOMWARE)
    theirs = harness.submit(RANSOMWARE, org_id=other_org, kb_id=other_kb)
    worker = harness.worker()
    assert worker.run_once() and worker.run_once()

    own = harness.search("ransomware escalation")
    assert {r.document_id for r in own.results} == {mine.document_id}
    foreign = harness.search("ransomware escalation", org_id=other_org, kb_ids=[other_kb])
    assert {r.document_id for r in foreign.results} == {theirs.document_id}
    with pytest.raises(SearchScopeError):  # my org, their KB
        harness.search("ransomware", kb_ids=[other_kb])
    with pytest.raises(SearchScopeError):
        harness.search("ransomware", kb_ids=[harness.kb.id, other_kb])


def test_forged_vector_payload_cannot_leak_another_tenants_chunk(harness: Harness) -> None:
    """Defense in depth: even if Qdrant payload filters were wrong, PostgreSQL re-checks."""
    other_org, other_kb = harness.add_tenant("other")
    harness.submit(BUDGET)
    theirs = harness.submit(RANSOMWARE, org_id=other_org, kb_id=other_kb)
    worker = harness.worker()
    assert worker.run_once() and worker.run_once()
    with harness.sessions() as session:
        chunk = session.scalar(
            select(DocumentChunk).where(DocumentChunk.document_id == theirs.document_id)
        )
        index = session.get(IndexVersion, harness.index_id)
        model = session.get(EmbeddingModel, index.embedding_model_id)
    vectors = harness.vectors(index, model)
    vectors.upsert(
        [
            ChunkPoint(
                id=chunk.id,  # their chunk id, relabelled with my tenant's payload
                vector=harness.embedder.embed_query("ransomware escalation lead"),
                org_id=harness.org.id,
                kb_id=harness.kb.id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                document_index_id=chunk.document_index_id,
                chunk_index=chunk.chunk_index,
            )
        ]
    )
    vectors.set_published(chunk.document_index_id, True)
    result = harness.search("ransomware escalation lead")
    assert result.candidates >= 1 and result.dropped_by_recheck >= 1
    assert theirs.document_id not in {r.document_id for r in result.results}


def test_search_requires_explicit_scope(harness: Harness) -> None:
    with pytest.raises(SearchScopeError, match="at least one knowledge base"):
        dense_search(
            harness.sessions,
            lambda _: harness.embedder,
            harness.vectors,
            org_id=harness.org.id,
            kb_ids=[],
            query="anything",
            index_version_id=harness.index_id,
        )
    with pytest.raises(SearchScopeError, match="no active index"):
        dense_search(
            harness.sessions,
            lambda _: harness.embedder,
            harness.vectors,
            org_id=harness.org.id,
            kb_ids=[harness.kb.id],
            query="anything",
        )
    with harness.sessions.begin() as session:
        activate_index(session, harness.index_id)
    assert (
        dense_search(
            harness.sessions,
            lambda _: harness.embedder,
            harness.vectors,
            org_id=harness.org.id,
            kb_ids=[harness.kb.id],
            query="anything",
        ).results
        == []
    )


def test_activating_a_second_index_demotes_the_first(harness: Harness) -> None:
    with harness.sessions.begin() as session:
        activate_index(session, harness.index_id)
        other = create_index(
            session,
            model_name="regex-v1",
            revision="test-v1",
            dimension=384,
            max_seq_tokens=256,
            chunking=ChunkingConfig(max_tokens=128),
        )
    with harness.sessions.begin() as session:
        assert activate_index(session, other.id)["previous"] == str(harness.index_id)
    with harness.sessions() as session:
        assert session.get(IndexVersion, harness.index_id).status == "building"
        assert session.get(IndexVersion, other.id).status == "active"


# ------------------------------------------------- Phase 2 processed documents


def test_phase2_processed_jobs_can_be_requeued_and_published_without_reparse(
    harness: Harness,
) -> None:
    submission = harness.submit(RANSOMWARE)
    # Reproduce the exact Phase 2 end state: chunks persisted, job completed/processed.
    harness.worker(vector_index_factory=flaky(harness, ensure_collection=1)).run_once()
    with harness.sessions.begin() as session:
        job = session.get(IngestionJob, submission.job_id)
        job.status, job.stage, job.finished_at = "completed", "processed", utcnow()
        job.error_code = job.error_message = None
    assert not harness.worker().run_once()  # completed jobs are never claimed by themselves
    with harness.sessions.begin() as session:
        assert requeue_processed(session, harness.org.id) == 1
    with harness.sessions.begin() as session:
        assert requeue_processed(session, harness.org.id) == 0  # idempotent

    class MustNotReparse(InlineProcessor):
        def process(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("processed chunks should be reused")

    assert harness.worker(MustNotReparse()).run_once()
    with harness.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.stage, job.attempt) == ("completed", "published", 1)
    assert harness.search("ransomware").results
