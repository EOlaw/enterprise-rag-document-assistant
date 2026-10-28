"""BM25 sparse vectors, hybrid fusion, reranking, replenishment and maintenance commands.

Same backends as test_indexing: SQLite/PostgreSQL and local/real Qdrant.
"""

from uuid import UUID

import pytest
from kip.cli import activate_index
from kip.db.models import (
    Document,
    DocumentChunk,
    DocumentIndex,
    DocumentVersion,
    EmbeddingModel,
    IndexVersion,
    IngestionJob,
    utcnow,
)
from kip.maintenance import (
    enqueue_reindex,
    mark_obsolete,
    missing_from_index,
    purge_vectors,
    retry_failed,
)
from kip.retrieval import SearchScopeError
from qdrant_client import models
from sqlalchemy import func, select

from rag.indexing import ChunkPoint
from rag.reranking import RerankInputTooLongError
from rag.retrieval import FusionConfig
from rag.sparse import Bm25Config, encode_document
from tests.fakes import OverlapReranker
from tests.integration.conftest import Harness
from tests.integration.test_indexing import flaky, make_due

POLICY = b"Control RC.RP-01 requires executing the recovery portion of the incident response plan."
RANSOMWARE = (
    b"Ransomware incidents must be escalated to the security operations lead within one hour."
)
BUDGET = b"The facilities team prepares the quarterly budget forecast every March."
PASSWORDS = b"Passwords used alone must be at least fifteen characters long."


@pytest.fixture
def hybrid(harness: Harness) -> Harness:
    """Harness whose default index version is a hybrid (dense + BM25) one."""
    harness.index_id = harness.hybrid_index()
    return harness


def publish(harness: Harness, *bodies: bytes) -> list[UUID]:
    documents = [harness.submit(body, filename=f"doc{i}.txt").document_id
                 for i, body in enumerate(bodies)]  # fmt: skip
    worker = harness.worker()
    while worker.run_once():
        pass
    return documents


def sparse_vectors(harness: Harness) -> dict[str, tuple[list[int], list[float]]]:
    points, _ = harness.client.scroll(
        harness.collection(), limit=1000, with_vectors=["bm25"], with_payload=False
    )
    out = {}
    for point in points:
        vector = point.vector["bm25"]
        out[str(point.id)] = (list(vector.indices), [round(v, 6) for v in vector.values])
    return out


# ------------------------------------------------------------------ publication


def test_hybrid_index_publishes_dense_and_bm25_vectors(hybrid: Harness) -> None:
    publish(hybrid, POLICY, BUDGET)
    stored = sparse_vectors(hybrid)
    with hybrid.sessions() as session:
        chunks = session.scalars(select(DocumentChunk)).all()
    assert len(stored) == len(chunks) == 2
    for chunk in chunks:
        expected = encode_document(chunk.embedding_text, Bm25Config())
        indices, values = stored[str(chunk.id)]
        assert indices == list(expected.indices)
        assert values == [round(v, 6) for v in expected.values]


def test_modes_share_scope_and_report_diagnostics(hybrid: Harness) -> None:
    policy, _ = publish(hybrid, POLICY, BUDGET)
    sparse = hybrid.search("RC.RP-01", mode="sparse")
    assert [r.document_id for r in sparse.results] == [policy]
    assert sparse.results[0].score_kind == "bm25"
    assert sparse.query_terms == 4  # "rc.rp-01" plus its parts rc, rp, 01
    combined = hybrid.search("RC.RP-01 recovery plan", mode="hybrid")
    top = combined.results[0]
    assert top.document_id == policy and top.score_kind == "rrf"
    assert top.dense_rank is not None and top.sparse_rank is not None
    k = combined.fusion["k"]
    assert top.fused_score == pytest.approx(1 / (k + top.dense_rank) + 1 / (k + top.sparse_rank))
    assert set(combined.retrievers) == {"dense", "sparse"}
    assert hybrid.search("RC.RP-01", mode="dense").results  # dense still works here


def test_sparse_and_hybrid_require_a_hybrid_index_version(harness: Harness) -> None:
    publish(harness, POLICY)
    for mode in ("sparse", "hybrid"):
        with pytest.raises(SearchScopeError, match="no BM25 vectors"):
            harness.search("RC.RP-01", mode=mode)


def test_stop_word_query_returns_nothing_from_bm25(hybrid: Harness) -> None:
    publish(hybrid, POLICY)
    result = hybrid.search("the and of", mode="sparse")
    assert result.results == [] and result.query_terms == 0


def test_retry_rewrites_identical_sparse_vectors(hybrid: Harness) -> None:
    publish(hybrid, POLICY)
    before = sparse_vectors(hybrid)
    with hybrid.sessions.begin() as session:
        session.scalar(select(DocumentIndex)).status = "building"
        job = session.scalar(select(IngestionJob))
        job.status, job.stage = "queued", "queued"
    assert hybrid.worker().run_once()
    assert sparse_vectors(hybrid) == before


def test_outage_during_hybrid_upsert_retries_and_stays_invisible(hybrid: Harness) -> None:
    submission = hybrid.submit(POLICY)
    assert hybrid.worker(vector_index_factory=flaky(hybrid, upsert=1)).run_once()
    assert hybrid.search("RC.RP-01", mode="sparse").results == []
    make_due(hybrid, submission.job_id)
    assert hybrid.worker().run_once()
    assert hybrid.search("RC.RP-01", mode="sparse").results


def test_dense_only_collection_cannot_serve_a_hybrid_index(hybrid: Harness) -> None:
    hybrid.client.create_collection(
        hybrid.collection(),
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    )
    submission = hybrid.submit(POLICY)
    assert hybrid.worker().run_once()
    with hybrid.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert (job.status, job.error_code) == ("failed", "vector_collection_mismatch")


# ---------------------------------------------------- isolation and visibility


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_tenant_isolation_with_identical_content(hybrid: Harness, mode: str) -> None:
    other_org, other_kb = hybrid.add_tenant("other")
    mine = hybrid.submit(POLICY)
    theirs = hybrid.submit(POLICY, org_id=other_org, kb_id=other_kb)
    worker = hybrid.worker()
    assert worker.run_once() and worker.run_once()
    own = hybrid.search("RC.RP-01 recovery", mode=mode)
    assert {r.document_id for r in own.results} == {mine.document_id}
    foreign = hybrid.search("RC.RP-01 recovery", mode=mode, org_id=other_org, kb_ids=[other_kb])
    assert {r.document_id for r in foreign.results} == {theirs.document_id}
    with pytest.raises(SearchScopeError):
        hybrid.search("RC.RP-01", mode=mode, kb_ids=[hybrid.kb.id, other_kb])


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_forged_sparse_payload_is_removed_by_recheck(hybrid: Harness, mode: str) -> None:
    other_org, other_kb = hybrid.add_tenant("other")
    hybrid.submit(BUDGET)
    theirs = hybrid.submit(POLICY, org_id=other_org, kb_id=other_kb)
    worker = hybrid.worker()
    assert worker.run_once() and worker.run_once()
    with hybrid.sessions() as session:
        chunk = session.scalar(
            select(DocumentChunk).where(DocumentChunk.document_id == theirs.document_id)
        )
        index = session.get(IndexVersion, hybrid.index_id)
        model = session.get(EmbeddingModel, index.embedding_model_id)
    vectors = hybrid.vectors(index, model)
    vectors.upsert(
        [
            ChunkPoint(
                id=chunk.id,  # their chunk, relabelled with my tenant's payload
                vector=hybrid.embedder.embed_query(chunk.embedding_text),
                org_id=hybrid.org.id,
                kb_id=hybrid.kb.id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                document_index_id=chunk.document_index_id,
                chunk_index=chunk.chunk_index,
                sparse=encode_document(chunk.embedding_text, Bm25Config()),
            )
        ]
    )
    vectors.set_published(chunk.document_index_id, True)
    result = hybrid.search("RC.RP-01 recovery", mode=mode)
    assert result.dropped_by_recheck >= 1
    assert theirs.document_id not in {r.document_id for r in result.results}


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_deletion_and_supersession_apply_to_every_mode(hybrid: Harness, mode: str) -> None:
    first = hybrid.submit(b"Incident records are retained for three years.")
    doomed = hybrid.submit(RANSOMWARE, filename="ransom.txt")
    worker = hybrid.worker()
    assert worker.run_once() and worker.run_once()
    hybrid.submit(b"Incident records are retained for seven years.", document_id=first.document_id)
    assert worker.run_once()
    with hybrid.sessions.begin() as session:
        session.get(Document, doomed.document_id).deleted_at = utcnow()
    texts = [r.text for r in hybrid.search("incident records retained", mode=mode).results]
    assert texts[0] == "Incident records are retained for seven years."
    assert "Incident records are retained for three years." not in texts
    ransom = hybrid.search("ransomware escalated", mode=mode)
    assert doomed.document_id not in {r.document_id for r in ransom.results}


# ----------------------------------------------------------------- replenishment


def test_replenishment_refills_after_recheck_drops_candidates(hybrid: Harness) -> None:
    # Six documents that BM25 ranks highest (more query-term occurrences) are deleted.
    strong = [f"Backup policy snapshots: backup snapshots policy {i}.".encode() for i in range(6)]
    weak = [f"Backup policy {i} for the archive team.".encode() for i in range(2)]
    documents = publish(hybrid, *strong, *weak)
    with hybrid.sessions.begin() as session:
        for document_id in documents[:6]:
            session.get(Document, document_id).deleted_at = utcnow()
    result = hybrid.search("backup policy snapshots", mode="sparse", top_k=2, overfetch=1)
    stats = result.retrievers["sparse"]
    assert len(result.results) == 2 and stats.rounds > 1
    assert {r.document_id for r in result.results} <= set(documents[6:])
    capped = hybrid.search(
        "backup policy snapshots", mode="sparse", top_k=2, overfetch=1, max_candidates=4
    )
    assert capped.retrievers["sparse"].capped and len(capped.results) < 2


def test_hybrid_fuses_only_visible_candidates(hybrid: Harness) -> None:
    documents = publish(hybrid, POLICY, b"RC.RP-01 superseded draft wording.", BUDGET)
    with hybrid.sessions.begin() as session:
        session.get(Document, documents[1]).deleted_at = utcnow()
    result = hybrid.search("RC.RP-01", mode="hybrid", explain=True)
    assert documents[1] not in {r.document_id for r in result.results}
    assert any(not item["visible"] for item in result.trace["sparse"])
    ranks = [r.sparse_rank for r in result.results if r.sparse_rank is not None]
    assert ranks == list(range(1, len(ranks) + 1))  # visible ranks are contiguous


# --------------------------------------------------------------------- reranking


def test_reranking_is_bounded_and_keeps_retrieval_ranks(hybrid: Harness) -> None:
    publish(hybrid, POLICY, RANSOMWARE, BUDGET, PASSWORDS)
    reranker = OverlapReranker()
    result = hybrid.search(
        "budget forecast March facilities",
        mode="hybrid",
        top_k=2,
        reranker=reranker,
        rerank_candidates=3,
    )
    assert reranker.calls == [3]
    assert result.results[0].text.startswith("The facilities team")
    assert result.results[0].retrieval_rank >= 1
    assert result.results[0].score_kind == "cross_encoder_logit"
    assert result.rerank["candidates"] == 3
    assert all(r.retrieval_rank >= 1 and r.rerank_score is not None for r in result.results)


def test_reranker_refuses_pairs_longer_than_its_window(hybrid: Harness) -> None:
    publish(hybrid, POLICY)
    with pytest.raises(RerankInputTooLongError):
        hybrid.search("RC.RP-01", mode="hybrid", reranker=OverlapReranker(max_length=8))


def test_fusion_configuration_is_explicit(hybrid: Harness) -> None:
    publish(hybrid, POLICY, BUDGET)
    result = hybrid.search("RC.RP-01", mode="hybrid", fusion=FusionConfig(k=10, sparse_weight=2))
    assert result.fusion["k"] == 10 and result.fusion["sparse_weight"] == 2


# ------------------------------------------------------------ upgrade and ops


def test_dense_to_hybrid_upgrade_path(harness: Harness) -> None:
    dense_index = harness.index_id
    publish(harness, POLICY, BUDGET)
    with harness.sessions.begin() as session:
        activate_index(session, dense_index)
    hybrid_index = harness.hybrid_index()
    with harness.sessions.begin() as session:
        assert missing_from_index(session, hybrid_index) == 2
        with pytest.raises(SystemExit, match="not ready in this index version"):
            activate_index(session, hybrid_index)
    with harness.sessions.begin() as session:
        assert enqueue_reindex(session, harness.org.id, hybrid_index) == 2
    with harness.sessions.begin() as session:
        assert enqueue_reindex(session, harness.org.id, hybrid_index) == 0  # idempotent
    worker = harness.worker()
    while worker.run_once():
        pass
    with harness.sessions.begin() as session:
        assert activate_index(session, hybrid_index)["missing_documents"] == 0
    assert harness.search("RC.RP-01", mode="sparse", index_id=hybrid_index).results
    # The dense index version remains searchable explicitly, for comparison or rollback.
    assert harness.search("RC.RP-01", mode="dense", index_id=dense_index).results


def test_cleanup_removes_obsolete_vectors_and_retries_failures(hybrid: Harness) -> None:
    first = hybrid.submit(b"Old retention guidance: three years.")
    doomed = hybrid.submit(RANSOMWARE, filename="ransom.txt")
    keep = hybrid.submit(BUDGET, filename="budget.txt")
    worker = hybrid.worker()
    while worker.run_once():
        pass
    hybrid.submit(b"New retention guidance: seven years.", document_id=first.document_id)
    assert worker.run_once()
    with hybrid.sessions.begin() as session:
        session.get(Document, doomed.document_id).deleted_at = utcnow()
    with hybrid.sessions.begin() as session:
        assert mark_obsolete(session, hybrid.org.id) == 2  # superseded v1 + deleted doc
    failing = purge_vectors(hybrid.sessions, flaky(hybrid, delete=1), hybrid.org.id)
    assert (failing.purged, failing.failed) == (1, 1)
    retried = purge_vectors(hybrid.sessions, hybrid.vectors, hybrid.org.id)
    assert (retried.purged, retried.failed) == (1, 0)
    payloads = [p["document_id"] for p in harness_payloads(hybrid)]
    assert str(doomed.document_id) not in payloads
    assert payloads.count(str(first.document_id)) == 1  # only the current version remains
    with hybrid.sessions() as session:
        statuses = sorted(i.status for i in session.scalars(select(DocumentIndex)))
        assert statuses == ["deleted", "deleted", "ready", "ready"]
    with hybrid.sessions.begin() as session:
        assert mark_obsolete(session, hybrid.org.id) == 0  # idempotent
    assert (
        hybrid.search("budget forecast", mode="hybrid").results[0].document_id == keep.document_id
    )


def harness_payloads(harness: Harness) -> list[dict]:
    points, _ = harness.client.scroll(harness.collection(), limit=1000, with_payload=True)
    return [p.payload for p in points]


def test_cleanup_skips_indexes_with_active_jobs(hybrid: Harness) -> None:
    submission = hybrid.submit(POLICY)
    assert hybrid.worker().run_once()
    hybrid.submit(b"A newer policy.", document_id=submission.document_id)
    with hybrid.sessions.begin() as session:
        session.get(Document, submission.document_id).deleted_at = utcnow()
        session.scalar(
            select(IngestionJob).where(IngestionJob.document_version_id == submission.version_id)
        ).status = "retrying"
    with hybrid.sessions.begin() as session:
        assert mark_obsolete(session, hybrid.org.id) == 0


def test_retry_failed_requeues_only_failed_jobs_in_scope(hybrid: Harness) -> None:
    other_org, other_kb = hybrid.add_tenant("other")
    failing = hybrid.submit(POLICY)
    for _ in range(3):
        hybrid.worker(vector_index_factory=flaky(hybrid, ensure_collection=9)).run_once()
        make_due(hybrid, failing.job_id)
    deleted = hybrid.submit(b"   ", filename="empty.txt")  # empty_document: terminal
    assert hybrid.worker().run_once()
    done = hybrid.submit(BUDGET, filename="budget.txt")
    assert hybrid.worker().run_once()
    with hybrid.sessions.begin() as session:
        session.get(Document, deleted.document_id).deleted_at = utcnow()
    with hybrid.sessions.begin() as session:
        assert retry_failed(session, other_org).retried == []  # other tenant: no effect
    with hybrid.sessions.begin() as session:
        outcome = retry_failed(session, hybrid.org.id, error_code="vector_store_unavailable")
    assert outcome.retried == [failing.job_id]
    with hybrid.sessions.begin() as session:
        outcome = retry_failed(session, hybrid.org.id)
    assert outcome.retried == [] and str(deleted.job_id) in outcome.skipped
    with hybrid.sessions() as session:
        job = session.get(IngestionJob, failing.job_id)
        assert (job.status, job.attempt, job.error_code) == ("queued", 0, None)
        assert session.get(IngestionJob, done.job_id).status == "completed"
        # Chunks were persisted before the vector failure, so the version stays processed.
        assert session.get(DocumentVersion, failing.version_id).status == "processed"
    assert hybrid.worker().run_once()
    with hybrid.sessions() as session:
        assert session.get(IngestionJob, failing.job_id).stage == "published"
    assert hybrid.search("RC.RP-01", mode="sparse").results
    assert other_kb  # second tenant existed throughout


def test_published_count_reconciles_with_postgres(hybrid: Harness) -> None:
    publish(hybrid, POLICY, BUDGET, PASSWORDS)
    with hybrid.sessions() as session:
        ready = session.scalar(
            select(func.sum(DocumentIndex.chunk_count)).where(DocumentIndex.status == "ready")
        )
    assert hybrid.client.count(hybrid.collection(), exact=True).count == ready


def test_due_retry_backoff_is_respected(hybrid: Harness) -> None:
    submission = hybrid.submit(POLICY)
    hybrid.worker(vector_index_factory=flaky(hybrid, upsert=1)).run_once()
    with hybrid.sessions() as session:
        job = session.get(IngestionJob, submission.job_id)
        assert job.status == "retrying"
    assert not hybrid.worker().run_once()  # not due yet
