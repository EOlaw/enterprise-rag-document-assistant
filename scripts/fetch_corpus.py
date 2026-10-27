"""Download an evaluation corpus and verify every file against its pinned SHA-256.

    python scripts/fetch_corpus.py evals/datasets/nist-security/v1/corpus.json

Files go to data/corpus/<name>/ (git-ignored). Existing files are re-verified, not
re-downloaded. A hash mismatch is an error: the publisher changed the file, so results
would no longer be comparable with earlier evaluation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import httpx

MAX_BYTES = 50 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=60) as response,
        tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as scratch,
    ):
        response.raise_for_status()
        size = 0
        for block in response.iter_bytes():
            size += len(block)
            if size > MAX_BYTES:
                raise ValueError(f"{url} exceeds {MAX_BYTES} bytes")
            scratch.write(block)
    Path(scratch.name).replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folder = args.dest / manifest["name"]
    failures = 0
    for document in manifest["documents"]:
        target = folder / document["file"]
        if not target.exists():
            fetch(document["url"], target)
        actual = sha256_file(target)
        status = "ok" if actual == document["sha256"] else "SHA256 MISMATCH"
        failures += status != "ok"
        print(json.dumps({"key": document["key"], "file": str(target), "status": status}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
