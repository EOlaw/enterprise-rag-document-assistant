"""Evaluate dense retrieval end to end (embedder -> Qdrant -> PostgreSQL re-check).

    python scripts/evaluate_retrieval.py evals/datasets/nist-security/v1 \
        --org <org_id> --kb <kb_id> [--index <index_version_id>] [--foreign-org <org_id>]

The corpus must already be ingested and published in that organization/KB (see
evals/datasets/nist-security/README.md). Documents are matched to dataset keys by the
pinned SHA-256 in corpus.json, and labels are resolved against the *stored* canonical
text of the published version, so a parser change that moves text fails loudly.

Writes evals/reports/<run_id>/report.json and report.md. Metrics are reported with 95%
bootstrap intervals over questions; with ~40 questions the intervals are wide and small
differences between configurations are not meaningful.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from kip.db.models import Document, DocumentVersion, EmbeddingModel, IndexVersion
from kip.db.session import make_engine, session_factory
from kip.retrieval import dense_search, resolve_index
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import EmbedderCache, QdrantIndexFactory
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from rag.evaluation.dataset import Dataset, load_dataset, resolve_spans
from rag.evaluation.metrics import bootstrap, query_metrics
from rag.evaluation.relevance import RetrievedSpan

KS = (1, 3, 5, 10)
METRICS = ("recall", "precision", "hit", "reciprocal_rank", "ndcg")


def published_documents(
    sessions: sessionmaker[Session], dataset: Dataset, org_id: UUID, kb_id: UUID
) -> dict[str, DocumentVersion]:
    wanted = {doc["sha256"]: doc["key"] for doc in dataset.manifest["documents"]}
    with sessions() as session:
        rows = session.scalars(
            select(DocumentVersion)
            .join(Document, Document.current_version_id == DocumentVersion.id)
            .where(
                Document.org_id == org_id,
                Document.kb_id == kb_id,
                Document.deleted_at.is_(None),
                DocumentVersion.status == "ready",
                DocumentVersion.raw_sha256.in_(wanted),
            )
        ).all()
    found = {wanted[row.raw_sha256]: row for row in rows}
    missing = sorted(set(wanted.values()) - set(found))
    if missing:
        raise SystemExit(f"documents not published in this org/KB: {missing}")
    return found


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def environment(settings: Settings, model: EmbeddingModel, index: IndexVersion) -> dict[str, Any]:
    try:
        server = httpx.get(settings.qdrant_url, timeout=5).json().get("version")
    except (httpx.HTTPError, ValueError):
        server = None
    packages = ["torch", "sentence-transformers", "transformers", "qdrant-client", "pypdf"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_only": settings.embed_device == "cpu",
        "packages": {name: package_version(name) for name in packages},
        "qdrant_server": server,
        "embedding_model": model.name,
        "embedding_revision": model.revision,
        "dimension": model.dimension,
        "max_seq_tokens": model.max_seq_tokens,
        "index_version_id": str(index.id),
        "index_fingerprint": index.fingerprint,
        "parser_version": index.parser_version,
        "chunking": index.chunking_config,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--org", type=UUID, required=True)
    parser.add_argument("--kb", type=UUID, required=True)
    parser.add_argument("--index", type=UUID)
    parser.add_argument(
        "--foreign-org", type=UUID, help="Tenant whose documents must never appear."
    )
    parser.add_argument("--out", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    settings = Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    sessions = session_factory(engine)
    store = LocalObjectStore(settings.storage_root)
    embedders = EmbedderCache(settings)
    vectors = QdrantIndexFactory(settings)
    try:
        with sessions() as session:
            index = resolve_index(session, args.index)
            model = session.get(EmbeddingModel, index.embedding_model_id)
            assert model is not None
            foreign: set[UUID] = set()
            if args.foreign_org:
                foreign = set(
                    session.scalars(select(Document.id).where(Document.org_id == args.foreign_org))
                )
        versions = published_documents(sessions, dataset, args.org, args.kb)
        key_by_document = {version.document_id: key for key, version in versions.items()}
        canonical = {
            key: store.get(version.canonical_text_key or "", max_bytes=50_000_000).decode("utf-8")
            for key, version in versions.items()
        }
        loaded = time.perf_counter()
        embedders(model)  # load once; model load time is reported separately
        model_load_s = time.perf_counter() - loaded

        def search(query: str) -> Any:
            return dense_search(
                sessions,
                embedders,
                vectors,
                org_id=args.org,
                kb_ids=[args.kb],
                query=query,
                top_k=max(KS),
                index_version_id=index.id,
                overfetch=settings.search_overfetch,
            )

        search("warm-up query, not measured")
        rows: list[dict[str, Any]] = []
        per_k: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in KS}
        latency: dict[str, list[float]] = defaultdict(list)
        leaks = 0
        for question in dataset.questions:
            spans = resolve_spans(question, canonical)
            result = search(question.question)
            ranked = [
                RetrievedSpan(key_by_document.get(r.document_id, "?"), r.char_start, r.char_end)
                for r in result.results
            ]
            leaks += sum(1 for r in result.results if r.document_id in foreign)
            for name, value in result.timings_ms.items():
                latency[name].append(value)
            row: dict[str, Any] = {"id": question.id, "category": question.category}
            for k in KS:
                metrics = query_metrics(spans, ranked, k)
                for metric in METRICS:
                    per_k[k][metric].append(getattr(metrics, metric))
                row[f"recall@{k}"] = metrics.recall
            row["rr@10"] = per_k[10]["reciprocal_rank"][-1]
            if question.outdated:
                row["outdated_in_top5"] = sum(
                    1 for r in ranked[:5] if r.document in question.outdated
                )
            row["top5"] = [
                {"doc": span.document, "chunk": r.chunk_index, "pages": [r.page_start, r.page_end],
                 "score": round(r.score, 4)}
                for span, r in zip(ranked[:5], result.results[:5], strict=True)
            ]  # fmt: skip
            row["dropped_by_recheck"] = result.dropped_by_recheck
            rows.append(row)

        summary = {
            f"{metric}@{k}": asdict(bootstrap(per_k[k][metric], seed=k))
            for k in KS
            for metric in METRICS
        }
        categories: dict[str, dict[str, float]] = {}
        for category in sorted({q.category for q in dataset.questions}):
            members = [i for i, q in enumerate(dataset.questions) if q.category == category]
            categories[category] = {
                "n": len(members),
                "recall@5": statistics.fmean(per_k[5]["recall"][i] for i in members),
                "mrr@10": statistics.fmean(per_k[10]["reciprocal_rank"][i] for i in members),
            }
        version_rows = [row for row in rows if "outdated_in_top5" in row]
        report = {
            "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-dense-"
            + index.fingerprint[:8],
            "created_at": datetime.now(UTC).isoformat(),
            "retriever": "dense (cosine, Qdrant) + PostgreSQL re-check; no reranking",
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "questions": len(dataset.questions),
                "questions_sha256": dataset.questions_sha256,
                "corpus_manifest_sha256": dataset.manifest_sha256,
                "documents": {key: str(v.document_id) for key, v in sorted(versions.items())},
            },
            "environment": environment(settings, model, index),
            "metrics": summary,
            "by_category": categories,
            "version_sensitivity": {
                "questions": len(version_rows),
                "with_outdated_doc_in_top5": sum(1 for r in version_rows if r["outdated_in_top5"]),
            },
            "isolation": {
                "foreign_org_checked": bool(args.foreign_org),
                "foreign_documents": len(foreign),
                "foreign_results": leaks,
            },
            "latency_ms": {
                name: {"p50": round(percentile(v, 0.5), 2), "p95": round(percentile(v, 0.95), 2)}
                for name, v in latency.items()
            },
            "model_load_seconds": round(model_load_s, 2),
            "questions": rows,
        }
    finally:
        engine.dispose()

    folder = args.out / report["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (folder / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(folder), "metrics@5": {
        m: round(summary[f"{m}@5"]["mean"], 3) for m in METRICS}}, indent=2))  # fmt: skip
    return 1 if leaks else 0


def render_markdown(report: dict[str, Any]) -> str:
    env, data = report["environment"], report["dataset"]
    lines = [
        f"# Dense retrieval evaluation `{report['run_id']}`",
        "",
        f"- Retriever: {report['retriever']}",
        f"- Dataset: `{data['name']}` v{data['version']}, {data['questions']} questions "
        f"(questions sha256 `{data['questions_sha256'][:12]}`, corpus manifest "
        f"`{data['corpus_manifest_sha256'][:12]}`)",
        f"- Model: `{env['embedding_model']}` @ `{env['embedding_revision'][:12]}`, "
        f"{env['dimension']}-d, max {env['max_seq_tokens']} tokens",
        f"- Index fingerprint `{env['index_fingerprint'][:16]}`, parser v{env['parser_version']}, "
        f"chunking {json.dumps(env['chunking'])}",
        f"- Environment: Python {env['python']}, {env['platform']}, CPU only: {env['cpu_only']}, "
        f"Qdrant {env['qdrant_server']}",
        "",
        "| Metric | @1 | @3 | @5 | @10 |",
        "|---|---|---|---|---|",
    ]
    for metric in METRICS:
        cells = []
        for k in KS:
            e = report["metrics"][f"{metric}@{k}"]
            cells.append(f"{e['mean']:.3f} [{e['ci_low']:.2f}, {e['ci_high']:.2f}]")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += ["", "95% percentile bootstrap intervals over questions (1,000 resamples).", "",
              "| Category | n | Recall@5 | MRR@10 |", "|---|---|---|---|"]  # fmt: skip
    for name, values in report["by_category"].items():
        lines.append(
            f"| {name} | {values['n']} | {values['recall@5']:.3f} | {values['mrr@10']:.3f} |"
        )
    version, isolation = report["version_sensitivity"], report["isolation"]
    lines += [
        "",
        f"- Version-sensitive questions with a superseded document in the top 5: "
        f"{version['with_outdated_doc_in_top5']} / {version['questions']}",
        f"- Cross-tenant check: {isolation['foreign_results']} results from "
        f"{isolation['foreign_documents']} foreign-tenant documents "
        f"(checked: {isolation['foreign_org_checked']})",
        "",
        "| Latency (ms) | p50 | p95 |",
        "|---|---|---|",
        *[f"| {n} | {v['p50']} | {v['p95']} |" for n, v in report["latency_ms"].items()],
        "",
        f"Model load (excluded from latency): {report['model_load_seconds']} s.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
