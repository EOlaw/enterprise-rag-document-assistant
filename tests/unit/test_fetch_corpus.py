from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_corpus.py"
GOOD = b"%PDF-1.4 public document"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_corpus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path: Path, digest: str) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"name": "demo", "documents": [
        {"key": "doc", "file": "doc.pdf", "url": "https://example.test/doc.pdf", "sha256": digest}
    ]}), encoding="utf-8")  # fmt: skip
    return path


def client(body: bytes, requests: list[str], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_downloads_verifies_and_does_not_redownload(tmp_path: Path, capsys) -> None:
    module, requests = load(), []
    path = manifest(tmp_path, hashlib.sha256(GOOD).hexdigest())
    dest = tmp_path / "corpus"
    assert module.main([str(path), "--dest", str(dest)], client=client(GOOD, requests)) == 0
    assert (dest / "demo" / "doc.pdf").read_bytes() == GOOD
    assert module.main([str(path), "--dest", str(dest)], client=client(GOOD, requests)) == 0
    assert requests == ["https://example.test/doc.pdf"]  # second run re-verified locally
    assert capsys.readouterr().out.count('"status": "ok"') == 2


def test_hash_mismatch_fails(tmp_path: Path, capsys) -> None:
    module = load()
    path = manifest(tmp_path, "0" * 64)
    assert module.main([str(path), "--dest", str(tmp_path / "c")], client=client(GOOD, [])) == 1
    assert "SHA256 MISMATCH" in capsys.readouterr().out


def test_oversized_or_failed_downloads_leave_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    module = load()
    monkeypatch.setattr(module, "MAX_BYTES", 4)
    path = manifest(tmp_path, hashlib.sha256(GOOD).hexdigest())
    dest = tmp_path / "c"
    with pytest.raises(ValueError, match="exceeds"):
        module.main([str(path), "--dest", str(dest)], client=client(GOOD, []))
    with pytest.raises(httpx.HTTPStatusError):
        module.main([str(path), "--dest", str(dest)], client=client(b"", [], status=404))
    assert list((dest / "demo").iterdir()) == []
