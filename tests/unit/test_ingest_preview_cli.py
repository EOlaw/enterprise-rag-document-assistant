from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest_preview.py"


def load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ingest_preview", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reports_chunks_and_typed_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "runbook.md"
    good.write_text("# Runbook\n\nRestart the service. Check ERR_CONN_RESET.\n", encoding="utf-8")
    bad = tmp_path / "fake.pdf"
    bad.write_bytes(b"not a pdf")

    exit_code = load_cli().main([str(good), str(bad), "--approx-tokens", "--show", "1"])

    reports = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert reports[0]["title"] == "Runbook"
    assert reports[0]["chunks"] == 1
    assert reports[0]["sample_chunks"][0]["heading_path"] == ["Runbook"]
    assert reports[1] == {
        "file": "fake.pdf",
        "error": "content_type_mismatch",
        "message": "file has a .pdf extension but no PDF header",
    }
