import json
import sys
from pathlib import Path

from kip import cli
from kip.db.models import DocumentIndex
from kip.worker import __main__ as worker_cli
from kip.worker.runner import Worker
from sqlalchemy import select

from tests.integration.conftest import Harness, InlineProcessor


def test_cli_bootstrap_submit_process_and_inspect(
    harness: Harness, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(
        "KIP_DATABASE_URL",
        harness.sessions.kw["bind"].url.render_as_string(
            hide_password=False,
        ),
    )
    # PostgreSQL tests use an isolated search_path; reuse that engine in both CLIs.
    monkeypatch.setattr(cli, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setattr(worker_cli, "make_engine", lambda _: harness.sessions.kw["bind"])
    monkeypatch.setenv("KIP_STORAGE_ROOT", str(harness.store.root))

    def invoke(*arguments: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["kip", *arguments])
        cli.main()
        return json.loads(capsys.readouterr().out)

    bootstrap = ("bootstrap", "--revision", "test-v1", "--model", "regex-v1")
    target = invoke(*bootstrap)
    assert invoke(*bootstrap) == target
    document = tmp_path / "guide.txt"
    document.write_text("Useful evidence for the CLI roundtrip.", encoding="utf-8")
    submission = invoke(
        "submit",
        str(document),
        "--org",
        target["org_id"],
        "--kb",
        target["kb_id"],
        "--index",
        target["index_version_id"],
    )
    assert not submission["duplicate"]
    assert invoke("job", submission["job_id"], "--org", target["org_id"])["status"] == "queued"
    monkeypatch.setattr(Worker, "_processor", lambda self, model: InlineProcessor())
    monkeypatch.setattr(worker_cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["kip-worker", "--once"])
    worker_cli.main()
    result = invoke("job", submission["job_id"], "--org", target["org_id"])
    assert result["status"] == "completed" and result["stage"] == "processed"
    with harness.sessions() as session:
        assert session.scalar(select(DocumentIndex)).status == "building"
