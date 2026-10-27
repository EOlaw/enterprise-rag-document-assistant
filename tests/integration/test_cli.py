import json
import sys
from pathlib import Path

import pytest
from kip import cli
from kip.db.models import DocumentIndex, IngestionJob
from kip.worker import __main__ as worker_cli
from kip.worker import runner
from kip.worker.runner import Worker
from sqlalchemy import select

from tests.integration.conftest import Harness, InlineProcessor


def test_cli_bootstrap_submit_index_activate_and_search(
    harness: Harness, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(
        "KIP_DATABASE_URL",
        harness.sessions.kw["bind"].url.render_as_string(hide_password=False),
    )
    # PostgreSQL tests use an isolated search_path; reuse that engine in both CLIs.
    monkeypatch.setattr(cli, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setattr(worker_cli, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setenv("KIP_STORAGE_ROOT", str(harness.store.root))
    # Real factories need the MiniLM model and a Qdrant server; use the test doubles.
    for module in (cli, runner):
        monkeypatch.setattr(module, "EmbedderCache", lambda _settings: lambda _m: harness.embedder)
        monkeypatch.setattr(module, "QdrantIndexFactory", lambda _settings: harness.vectors)
    monkeypatch.setattr(Worker, "_processor", lambda self, model: InlineProcessor())
    monkeypatch.setattr(worker_cli.signal, "signal", lambda *args: None)

    def invoke(*arguments: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["kip", *arguments])
        cli.main()
        return json.loads(capsys.readouterr().out)

    def work(*arguments: str) -> None:
        monkeypatch.setattr(sys, "argv", ["kip-worker", *arguments])
        worker_cli.main()

    bootstrap = ("bootstrap", "--revision", "test-v1", "--model", "regex-v1")
    target = invoke(*bootstrap)
    assert invoke(*bootstrap) == target
    org, kb, index = target["org_id"], target["kb_id"], target["index_version_id"]
    for name, body in [
        ("guide.txt", "Escalate ransomware incidents to the security operations lead."),
        ("other.txt", "Quarterly budget planning for the facilities team."),
    ]:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        submission = invoke("submit", str(path), "--org", org, "--kb", kb, "--index", index)
        assert invoke("job", submission["job_id"], "--org", org)["status"] == "queued"
    work("--until-idle")
    job = invoke("job", submission["job_id"], "--org", org)
    assert job["status"] == "completed" and job["stage"] == "published"

    search = ("search", "ransomware escalation lead", "--org", org, "--kb", kb)
    with pytest.raises(SystemExit, match="no active index"):
        invoke(*search)
    assert invoke("activate-index", index)["active"] == index
    found = invoke(*search, "--top-k", "1")
    assert [r["filename"] for r in found["results"]] == ["guide.txt"]
    assert found["results"][0]["text"].startswith("Escalate ransomware")
    with pytest.raises(SystemExit, match="knowledge base not found"):
        invoke("search", "x", "--org", org, "--kb", "00000000-0000-0000-0000-000000000000")
    with harness.sessions() as session:
        assert {i.status for i in session.scalars(select(DocumentIndex))} == {"ready"}
        assert {j.stage for j in session.scalars(select(IngestionJob))} == {"published"}
    assert invoke("requeue-processed", "--org", org) == {"requeued": 0}
