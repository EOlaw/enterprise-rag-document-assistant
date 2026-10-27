"""Run the document-processing pipeline on local files and print a JSON report.

Used to inspect parsing/chunking quality before anything is indexed:

    python scripts/ingest_preview.py data/corpus/nist/NIST.SP.800-61r2.pdf --show 3

Exit code is 0 when every file processed, 1 when any file failed (errors are reported
with their stable error code, never with document content).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rag.chunking.tokens import HuggingFaceTokenCounter, RegexTokenCounter, TokenCounter
from rag.config import ChunkingConfig
from rag.errors import IngestionError
from rag.ingestion.pipeline import ProcessedDocument, process_document

DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_SEQ_LENGTH = 256


def build_counter(args: argparse.Namespace) -> TokenCounter:
    if args.approx_tokens:
        return RegexTokenCounter()
    return HuggingFaceTokenCounter.from_pretrained(args.tokenizer, args.max_seq_length)


def summarize(doc: ProcessedDocument, elapsed_ms: float, show: int) -> dict[str, Any]:
    counts = sorted(c.token_count for c in doc.chunks)
    p95 = counts[min(len(counts) - 1, int(0.95 * len(counts)))]
    return {
        "file": doc.filename,
        "type": doc.file_type.value,
        "title": doc.title,
        "bytes": doc.byte_size,
        "pages": doc.page_count,
        "raw_sha256": doc.raw_sha256,
        "text_sha256": doc.text_sha256,
        "canonical_chars": len(doc.canonical_text),
        "blocks": len(doc.blocks),
        "chunks": len(doc.chunks),
        "tokens": {
            "tokenizer": doc.tokenizer,
            "min": counts[0],
            "mean": round(statistics.fmean(counts), 1),
            "p95": p95,
            "max": counts[-1],
            "total": sum(counts),
        },
        "chunking_fingerprint": doc.chunking_fingerprint,
        "parser_version": doc.parser_version,
        "processing_ms": round(elapsed_ms, 1),
        "warnings": doc.warnings,
        "sample_chunks": [
            {
                "index": c.index,
                "pages": [c.page_start, c.page_end],
                "heading_path": list(c.heading_path),
                "tokens": c.token_count,
                "chars": [c.char_start, c.char_end],
                "text": c.text[:400] + ("..." if len(c.text) > 400 else ""),
            }
            for c in doc.chunks[:show]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--strategy", choices=["structure_aware", "fixed_tokens"], default="structure_aware"
    )
    parser.add_argument("--max-tokens", type=int, default=ChunkingConfig().max_tokens)
    parser.add_argument("--overlap-tokens", type=int, default=ChunkingConfig().overlap_tokens)
    parser.add_argument("--page-boundaries", action="store_true")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--approx-tokens", action="store_true", help="regex counter (offline, inexact)"
    )
    parser.add_argument("--show", type=int, default=2, help="number of sample chunks to print")
    args = parser.parse_args(argv)

    config = ChunkingConfig(
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        page_boundaries=args.page_boundaries,
    )
    counter = build_counter(args)

    reports: list[dict[str, Any]] = []
    failed = False
    for path in args.paths:
        started = time.perf_counter()
        try:
            doc = process_document(
                path.read_bytes(), path.name, chunking=config, token_counter=counter
            )
        except IngestionError as exc:
            failed = True
            reports.append({"file": path.name, "error": exc.code, "message": exc.message})
            continue
        reports.append(summarize(doc, (time.perf_counter() - started) * 1000, args.show))

    json.dump(reports, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
