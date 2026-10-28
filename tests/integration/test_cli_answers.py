import json
import sys
from datetime import date
from uuid import UUID

import pytest
from kip import cli
from kip.answer_evaluation import (
    EvalItem,
    calibrate_sufficiency,
    run_answer_evaluation,
)
from kip.answering import AnswerSettings
from kip.db.models import DocumentVersion
from kip.publications import set_publication
from sqlalchemy import select

from rag.answering import AuthorityPolicy, PublicationStatus
from tests.fakes import OverlapReranker, ScriptedProvider, answer_json
from tests.integration.conftest import Harness

OLD = b"Incident evidence is retained for three years under the old guide."
NEW = b"Incident evidence is retained for seven years under the current guide."


def publish(harness: Harness) -> list[str]:
    harness.index_id = harness.hybrid_index()
    ids = [
        str(harness.submit(body, filename=name).document_id)
        for name, body in (("old.txt", OLD), ("new.txt", NEW))
    ]
    worker = harness.worker()
    while worker.run_once():
        pass
    return ids


@pytest.fixture
def invoke(harness: Harness, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setattr(cli, "EmbedderCache", lambda _s: lambda _m: harness.embedder)
    monkeypatch.setattr(cli, "QdrantIndexFactory", lambda _s: harness.vectors)
    monkeypatch.setattr(cli, "RerankerCache", lambda _s: OverlapReranker)
    monkeypatch.setattr(cli, "make_provider", lambda _s: ScriptedProvider())
    monkeypatch.setenv("KIP_STORAGE_ROOT", str(harness.store.root))

    def run(*arguments: str):  # type: ignore[no-untyped-def]
        monkeypatch.setattr(sys, "argv", ["kip", *arguments])
        cli.main()
        return json.loads(capsys.readouterr().out)

    return run


def test_publication_documents_ask_and_answer_run_commands(harness: Harness, invoke) -> None:
    old, new = publish(harness)
    org, kb, index = str(harness.org.id), str(harness.kb.id), str(harness.index_id)
    record = invoke(
        "set-publication",
        "--org",
        org,
        "--document",
        old,
        "--status",
        "withdrawn",
        "--published-on",
        "2012-08-01",
        "--superseded-by",
        new,
        "--note",
        "notice p1",
    )
    assert (
        record["publication_status"] == "withdrawn" and record["superseded_by_document_id"] == new
    )
    invoke("set-publication", "--org", org, "--document", new, "--status", "current")
    listed = {d["document_id"]: d for d in invoke("documents", "--org", org, "--kb", kb)}
    assert listed[new]["publication_status"] == "current" and listed[new]["published_on"] is None

    result = invoke(
        "ask",
        "How long is incident evidence retained?",
        "--org",
        org,
        "--kb",
        kb,
        "--index",
        index,
        "--trace",
    )
    assert result["grounding"] == "grounded"
    assert result["citations"][0]["publication_status"] == "current"
    assert [e["treatment"] for e in result["evidence"]] == ["include", "demote"]
    assert result["retrieval"]["reranked"] is True
    stored = invoke("answer-run", result["run_id"], "--org", org)
    assert stored["answer"] == result["answer"] and stored["trace"]["citations"]

    excluded = invoke(
        "ask",
        "How long is incident evidence retained?",
        "--org",
        org,
        "--kb",
        kb,
        "--index",
        index,
        "--policy",
        "withdrawn=exclude",
        "--no-rerank",
        "--no-store",
    )
    assert excluded["run_id"] is None
    assert [s["reason"] for s in excluded["skipped"]] == ["excluded_by_policy"]

    with pytest.raises(SystemExit, match="answer run not found"):
        invoke("answer-run", result["run_id"], "--org", "00000000-0000-0000-0000-000000000000")
    with pytest.raises(SystemExit, match="set-publication refused"):
        invoke(
            "set-publication",
            "--org",
            org,
            "--document",
            new,
            "--status",
            "current",
            "--superseded-by",
            old,
        )


def test_calibration_prefers_permissive_thresholds_on_ties() -> None:
    signals = [(True, 0.9, 5.0), (True, 0.8, 1.0), (False, 0.1, -8.0), (False, 0.6, -5.0)]
    result = calibrate_sufficiency(signals)
    assert result["balanced_accuracy"] == 1.0
    assert result["min_rerank"] == -4.0  # lowest logit floor that separates both classes
    assert result["confusion"] == {"answerable/sufficient": 2, "unanswerable/insufficient": 2}
    coverage_only = calibrate_sufficiency([(True, 0.9, None), (False, 0.2, None)])
    assert coverage_only["min_coverage"] == 0.25 and coverage_only["balanced_accuracy"] == 1.0


def test_answer_evaluation_scores_items(harness: Harness) -> None:
    old, new = (UUID(value) for value in publish(harness))
    with harness.sessions() as session:
        versions = {
            key: session.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc))
            for key, doc in (("old", old), ("new", new))
        }
    with harness.sessions.begin() as session:
        set_publication(
            session,
            harness.org.id,
            versions["old"].document_id,
            status=PublicationStatus.WITHDRAWN,
            published_on=date(2012, 8, 1),
            superseded_by=versions["new"].document_id,
        )
        set_publication(
            session,
            harness.org.id,
            versions["new"].document_id,
            status=PublicationStatus.CURRENT,
            published_on=date(2025, 4, 3),
        )
    items = [
        EvalItem(
            "c1",
            "current_vs_withdrawn",
            "How long is incident evidence retained?",
            "kb",
            [["seven"]],
            [{"doc": "new", "quote": "retained for seven years"}],
            [{"doc": "old", "quote": "retained for three years"}],
            None,
            None,
        ),
        EvalItem(
            "u1",
            "unanswerable",
            "Which football team won the cup final?",
            "kb",
            [],
            [],
            [],
            None,
            None,
        ),
    ]
    report = run_answer_evaluation(
        sessions=harness.sessions,
        store=harness.store,
        embedder_factory=lambda _: harness.embedder,
        vector_index_factory=harness.vectors,
        reranker_factory=None,
        provider=ScriptedProvider(
            lambda m: answer_json(
                [("Incident evidence is retained for seven years under the current guide.", ["S1"])]
            )
        ),
        items=items,
        org_id=harness.org.id,
        kbs={"kb": [harness.kb.id]},
        versions=versions,
        settings=AnswerSettings(rerank=False, budget_tokens=600),
        policies={"default": AuthorityPolicy(), "include": AuthorityPolicy(withdrawn="include")},
        index_version_id=harness.index_id,
    )
    default = report["results"]["default"]
    c1 = next(r for r in default["items"] if r["id"] == "c1")
    assert c1["facts_correct"] and c1["evidence_has_gold"] and c1["citations_correct"] == 1
    assert c1["cites_current"] and not c1["cites_older"] and c1["older_in_evidence"]
    u1 = next(r for r in default["items"] if r["id"] == "u1")
    assert u1["abstained_retrieval"] and not u1["answered"]
    summary = default["summary"]
    assert (
        summary["abstained_on_unanswerable"] == 1.0
        and summary["facts_correct_on_answerable"] == 1.0
    )
    assert set(report["results"]) == {"default", "include"}
