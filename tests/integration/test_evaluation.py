"""The evaluation library and script, end to end through the real search path."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from kip.evaluation import DEFAULT_CONFIGS, RunConfig, run_evaluation, write_report

from rag.evaluation.dataset import load_dataset
from tests.fakes import OverlapReranker
from tests.integration.conftest import Harness

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_retrieval.py"

OLD = b"Old guide: lessons learned are shared only after recovery concludes."
NEW = b"New guide: lessons learned should be shared as soon as they are identified."
TOC = (
    b"Contents\n\nIntroduction . . . . . . . . 1\nScope . . . . . . . . 2\nTerms . . . . . . . . 3"
)
IDS = b"Control RC.RP-01 executes the recovery portion of the incident response plan."
DOCS = {"old": OLD, "new": NEW, "toc": TOC, "ids": IDS}


def dataset_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    documents = [
        {
            "key": key,
            "file": f"{key}.txt",
            "url": f"https://example.test/{key}",
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for key, body in DOCS.items()
    ]
    (folder / "corpus.json").write_text(
        json.dumps({"name": "mini", "version": "1", "documents": documents}), encoding="utf-8"
    )
    questions = [
        {
            "id": "q1",
            "category": "version_sensitive",
            "question": "When should lessons learned be shared?",
            "outdated": ["old"],
            "relevant": [
                {"doc": "new", "quote": "shared as soon as they are identified", "grade": 2}
            ],
        },
        {
            "id": "q2",
            "category": "identifier",
            "question": "RC.RP-01",
            "relevant": [
                {"doc": "ids", "quote": "Control RC.RP-01 executes the recovery", "grade": 2}
            ],
        },
    ]
    (folder / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions), encoding="utf-8"
    )
    (folder / "lineage.json").write_text(json.dumps({"newer": {"new": ["old"]}}), encoding="utf-8")
    return folder


def load_corpus(harness: Harness) -> tuple:
    harness.index_id = harness.hybrid_index()
    for key, body in DOCS.items():
        harness.submit(body, filename=f"{key}.txt")
    foreign_org, foreign_kb = harness.add_tenant("foreign")
    harness.submit(NEW, org_id=foreign_org, kb_id=foreign_kb, filename="new.txt")
    worker = harness.worker()
    while worker.run_once():
        pass
    return foreign_org, foreign_kb


def test_run_evaluation_reports_every_configuration(harness: Harness, tmp_path: Path) -> None:
    foreign_org, _ = load_corpus(harness)
    report = run_evaluation(
        sessions=harness.sessions,
        store=harness.store,
        embedder_factory=lambda _: harness.embedder,
        vector_index_factory=harness.vectors,
        reranker_factory=OverlapReranker,
        dataset=load_dataset(dataset_folder(tmp_path)),
        org_id=harness.org.id,
        kb_id=harness.kb.id,
        index_version_id=harness.index_id,
        foreign_org=foreign_org,
    )
    assert list(report["results"]) == [c.name for c in DEFAULT_CONFIGS]
    assert set(report["comparisons"]) == {
        "bm25 - dense",
        "hybrid - dense",
        "hybrid - bm25",
        "hybrid+rerank - dense",
        "hybrid+rerank - hybrid",
    }
    for result in report["results"].values():
        for estimate in result["metrics"].values():
            assert 0 <= estimate["ci_low"] <= estimate["mean"] <= estimate["ci_high"] <= 1
        older = result["older_publications"]
        assert older["questions"] == 1 and older["rows"][0]["older_documents"] == ["old"]
    # Dense always ranks all four chunks, so the dot-leader TOC chunk is in its top 5; BM25
    # shares no terms between the TOC and either question, so it never returns it.
    assert report["results"]["dense"]["table_of_contents"]["toc_results_in_top5"] == 2
    assert report["results"]["bm25"]["table_of_contents"]["toc_results_in_top5"] == 0
    bm25 = report["results"]["bm25"]["questions"]
    assert next(q for q in bm25 if q["id"] == "q2")["recall@1"] == 1.0  # exact identifier
    assert report["isolation"] == {
        "foreign_org_checked": True,
        "foreign_documents": 1,
        "foreign_results": 0,
    }
    assert report["reranker"]["model"] == "overlap-test"
    folder = write_report(report, tmp_path / "reports")
    markdown = (folder / "report.md").read_text(encoding="utf-8")
    assert "| hybrid+rerank |" in markdown and "Older publications" in markdown
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["run_id"]


def test_missing_corpus_documents_are_reported(harness: Harness, tmp_path: Path) -> None:
    harness.index_id = harness.hybrid_index()
    harness.submit(OLD, filename="old.txt")
    assert harness.worker().run_once()
    with pytest.raises(LookupError, match="not published"):
        run_evaluation(
            sessions=harness.sessions,
            store=harness.store,
            embedder_factory=lambda _: harness.embedder,
            vector_index_factory=harness.vectors,
            reranker_factory=None,
            dataset=load_dataset(dataset_folder(tmp_path)),
            org_id=harness.org.id,
            kb_id=harness.kb.id,
            index_version_id=harness.index_id,
            configs=[RunConfig("dense", "dense")],
        )


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_main_writes_report_and_flags_foreign_results(
    harness: Harness, tmp_path: Path, monkeypatch, capsys
) -> None:
    foreign_org, _ = load_corpus(harness)
    script = load_script()
    monkeypatch.setattr(script, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setattr(script, "EmbedderCache", lambda _s: lambda _m: harness.embedder)
    monkeypatch.setattr(script, "QdrantIndexFactory", lambda _s: harness.vectors)
    monkeypatch.setattr(script, "RerankerCache", lambda _s: OverlapReranker)
    monkeypatch.setattr(script, "qdrant_version", lambda _url: "test")
    monkeypatch.setenv("KIP_STORAGE_ROOT", str(harness.store.root))
    arguments = [
        str(dataset_folder(tmp_path)),
        "--org",
        str(harness.org.id),
        "--kb",
        str(harness.kb.id),
        "--index",
        str(harness.index_id),
        "--configs",
        "bm25,hybrid",
        "--out",
        str(tmp_path / "out"),
    ]
    assert script.main([*arguments, "--foreign-org", str(foreign_org)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["summary"]) == {"bm25", "hybrid"}
    assert Path(summary["report"], "report.md").exists()
    # Point the "foreign" check at our own tenant: every result now counts as a leak.
    assert script.main([*arguments, "--foreign-org", str(harness.org.id)]) == 1
    with pytest.raises(argparse.ArgumentTypeError):
        script.parse_configs("dense,unknown")
