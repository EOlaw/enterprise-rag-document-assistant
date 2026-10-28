"""End-to-end retrieval evaluation over the real search path (Qdrant + PostgreSQL re-check).

Used by ``scripts/evaluate_retrieval.py``; kept here so it can be tested with the
integration harness. One run evaluates several retrieval configurations on the same
questions and reports:

* Recall/Precision/Hit/MRR/nDCG at k in {1,3,5,10} with 95% bootstrap intervals;
* paired per-question differences against the first configuration (bootstrap CI);
* per-category Recall@5 and MRR@10;
* **older publications** (separate documents, from ``lineage.json``) retrieved for
  questions labelled only on their newer replacements, and whether they outrank the
  first relevant result;
* **table-of-contents** chunks in the top 5 (dot-leader heuristic);
* a cross-tenant check and per-stage latency.

No parameter is tuned here: every configuration uses the values it is given.
"""

import json
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kip.db.models import Document, DocumentVersion, EmbeddingModel, IndexVersion
from kip.retrieval import Mode, RetrievedChunk, resolve_index, search
from kip.storage import ObjectStore
from kip.vectors import EmbedderFactory, VectorIndexFactory
from rag.evaluation.dataset import Dataset, resolve_spans
from rag.evaluation.diagnostics import looks_like_table_of_contents
from rag.evaluation.metrics import bootstrap, query_metrics
from rag.evaluation.relevance import COVER_FRACTION, RetrievedSpan, overlap_fraction
from rag.reranking import Reranker
from rag.retrieval import FusionConfig

KS = (1, 3, 5, 10)
METRICS = ("recall", "precision", "hit", "reciprocal_rank", "ndcg")


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: Mode
    rerank: bool = False


DEFAULT_CONFIGS = (
    RunConfig("dense", "dense"),
    RunConfig("bm25", "sparse"),
    RunConfig("hybrid", "hybrid"),
    RunConfig("hybrid+rerank", "hybrid", rerank=True),
)


@dataclass(frozen=True)
class SearchSettings:
    overfetch: int = 4
    max_candidates: int = 200
    hybrid_candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30


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
        raise LookupError(f"documents not published in this org/KB: {missing}")
    return found


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def _versions(names: Sequence[str]) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = package_version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def environment(
    model: EmbeddingModel, index: IndexVersion, *, qdrant_server: str | None, device: str
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_only": device == "cpu",
        "packages": _versions(
            ["torch", "sentence-transformers", "transformers", "qdrant-client", "pypdf"]
        ),
        "qdrant_server": qdrant_server,
        "embedding_model": model.name,
        "embedding_revision": model.revision,
        "dimension": model.dimension,
        "max_seq_tokens": model.max_seq_tokens,
        "index_version_id": str(index.id),
        "index_fingerprint": index.fingerprint,
        "parser_version": index.parser_version,
        "chunking": index.chunking_config,
        "sparse": index.sparse_config,
    }


def run_evaluation(
    *,
    sessions: sessionmaker[Session],
    store: ObjectStore,
    embedder_factory: EmbedderFactory,
    vector_index_factory: VectorIndexFactory,
    reranker_factory: Callable[[], Reranker] | None,
    dataset: Dataset,
    org_id: UUID,
    kb_id: UUID,
    index_version_id: UUID | None = None,
    foreign_org: UUID | None = None,
    configs: Sequence[RunConfig] = DEFAULT_CONFIGS,
    search_settings: SearchSettings | None = None,
    qdrant_server: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if not configs:
        raise ValueError("at least one configuration is required")
    knobs = search_settings or SearchSettings()
    with sessions() as session:
        index = resolve_index(session, index_version_id)
        model = session.get(EmbeddingModel, index.embedding_model_id)
        if model is None:
            raise LookupError("index version has no embedding model")
        foreign: set[UUID] = set()
        if foreign_org is not None:
            foreign = set(
                session.scalars(select(Document.id).where(Document.org_id == foreign_org))
            )
    versions = published_documents(sessions, dataset, org_id, kb_id)
    key_by_document = {version.document_id: key for key, version in versions.items()}
    canonical = {
        key: store.get(version.canonical_text_key or "", max_bytes=50_000_000).decode("utf-8")
        for key, version in versions.items()
    }
    spans = {q.id: resolve_spans(q, canonical) for q in dataset.questions}

    load_seconds: dict[str, float] = {}
    moment = time.perf_counter()
    embedder_factory(model)  # loaded once; excluded from per-query latency
    load_seconds["embedder"] = round(time.perf_counter() - moment, 2)
    reranker: Reranker | None = None
    if any(c.rerank for c in configs):
        if reranker_factory is None:
            raise ValueError("a reranking configuration needs a reranker factory")
        moment = time.perf_counter()
        reranker = reranker_factory()
        load_seconds["reranker"] = round(time.perf_counter() - moment, 2)

    def run(config: RunConfig, query: str) -> Any:
        return search(
            sessions,
            embedder_factory,
            vector_index_factory,
            org_id=org_id,
            kb_ids=[kb_id],
            query=query,
            mode=config.mode,
            top_k=max(KS),
            index_version_id=index.id,
            overfetch=knobs.overfetch,
            max_candidates=knobs.max_candidates,
            hybrid_candidates=knobs.hybrid_candidates,
            fusion=FusionConfig(k=knobs.rrf_k),
            reranker=reranker if config.rerank else None,
            rerank_candidates=max(knobs.rerank_candidates, *KS),
        )

    lineage = dataset.lineage
    older_keys = {old for olds in lineage.values() for old in olds}
    results: dict[str, Any] = {}
    per_question: dict[str, dict[str, dict[str, float]]] = {}
    leaks = 0
    for config in configs:
        run(config, "warm-up query, not measured")
        per_k: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in KS}
        latency: dict[str, list[float]] = defaultdict(list)
        rows: list[dict[str, Any]] = []
        lineage_rows: list[dict[str, Any]] = []
        toc_results = toc_questions = 0
        for question in dataset.questions:
            outcome = run(config, question.question)
            ranked = [_span(r, key_by_document) for r in outcome.results]
            leaks += sum(1 for r in outcome.results if r.document_id in foreign)
            for name, value in outcome.timings_ms.items():
                latency[name].append(value)
            row: dict[str, Any] = {"id": question.id, "category": question.category}
            for k in KS:
                metrics = query_metrics(spans[question.id], ranked, k)
                for metric in METRICS:
                    per_k[k][metric].append(getattr(metrics, metric))
                row[f"recall@{k}"] = metrics.recall
            row["rr@10"] = per_k[10]["reciprocal_rank"][-1]
            top5 = outcome.results[:5]
            toc = [r.rank for r in top5 if looks_like_table_of_contents(r.text)]
            toc_results += len(toc)
            toc_questions += bool(toc)
            row["toc_ranks_top5"] = toc
            row["top5"] = [
                {
                    "doc": key_by_document.get(r.document_id, "?"),
                    "chunk": r.chunk_index,
                    "pages": [r.page_start, r.page_end],
                    "score": round(r.score, 4),
                    "dense_rank": r.dense_rank,
                    "sparse_rank": r.sparse_rank,
                    "retrieval_rank": r.retrieval_rank,
                }
                for r in top5
            ]
            row["dropped_by_recheck"] = outcome.dropped_by_recheck
            labelled = {label["doc"] for label in question.labels}
            older = {old for doc in labelled for old in lineage.get(doc, [])} - labelled
            if older:
                lineage_rows.append(_lineage_row(question.id, spans[question.id], ranked, older))
            rows.append(row)
        per_question[config.name] = {
            row["id"]: {
                "recall@5": row["recall@5"],
                "recall@10": row["recall@10"],
                "rr@10": row["rr@10"],
            }
            for row in rows
        }
        categories: dict[str, dict[str, float]] = {}
        for category in sorted({q.category for q in dataset.questions}):
            members = [i for i, q in enumerate(dataset.questions) if q.category == category]
            categories[category] = {
                "n": len(members),
                "recall@5": statistics.fmean(per_k[5]["recall"][i] for i in members),
                "mrr@10": statistics.fmean(per_k[10]["reciprocal_rank"][i] for i in members),
            }
        n_questions = len(dataset.questions)
        results[config.name] = {
            "mode": config.mode,
            "rerank": config.rerank,
            "metrics": {
                f"{metric}@{k}": asdict(bootstrap(per_k[k][metric], seed=k))
                for k in KS
                for metric in METRICS
            },
            "by_category": categories,
            "older_publications": {
                "questions": len(lineage_rows),
                "with_older_in_top5": sum(1 for r in lineage_rows if r["older_in_top5"]),
                "older_outranks_first_relevant": sum(
                    1 for r in lineage_rows if r["older_outranks_relevant"]
                ),
                "rows": lineage_rows,
            },
            "table_of_contents": {
                "questions_with_toc_in_top5": toc_questions,
                "toc_results_in_top5": toc_results,
                "share_of_top5_results": round(toc_results / (5 * n_questions), 4),
            },
            "latency_ms": {
                name: {"p50": round(percentile(v, 0.5), 2), "p95": round(percentile(v, 0.95), 2)}
                for name, v in latency.items()
            },
            "questions": rows,
        }

    # Paired per-question differences: each configuration against the first (baseline) and
    # against the configuration listed just before it (e.g. hybrid vs bm25, rerank vs hybrid).
    pairs: list[tuple[str, str]] = []
    for position, config in enumerate(configs[1:], start=1):
        pairs.append((config.name, configs[0].name))
        previous = configs[position - 1].name
        if previous != configs[0].name:
            pairs.append((config.name, previous))
    comparisons: dict[str, Any] = {}
    for candidate, reference in pairs:
        deltas: dict[str, Any] = {}
        for metric in ("recall@5", "recall@10", "rr@10"):
            diffs = [
                per_question[candidate][q.id][metric] - per_question[reference][q.id][metric]
                for q in dataset.questions
            ]
            deltas[metric] = {
                **asdict(bootstrap(diffs, seed=101)),
                "better": sum(1 for d in diffs if d > 0),
                "worse": sum(1 for d in diffs if d < 0),
                "same": sum(1 for d in diffs if d == 0),
            }
        comparisons[f"{candidate} - {reference}"] = deltas

    return {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-compare-"
        + index.fingerprint[:8],
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "questions": len(dataset.questions),
            "questions_sha256": dataset.questions_sha256,
            "corpus_manifest_sha256": dataset.manifest_sha256,
            "lineage": lineage,
            "documents": {key: str(v.document_id) for key, v in sorted(versions.items())},
        },
        "environment": environment(model, index, qdrant_server=qdrant_server, device=device),
        "search_settings": asdict(knobs),
        "reranker": (
            {
                "model": reranker.model_id,
                "revision": reranker.revision,
                "max_length": reranker.max_length,
            }
            if reranker
            else None
        ),
        "configs": [asdict(c) for c in configs],
        "results": results,
        "comparisons": comparisons,
        "isolation": {
            "foreign_org_checked": foreign_org is not None,
            "foreign_documents": len(foreign),
            "foreign_results": leaks,
        },
        "load_seconds": load_seconds,
        "older_publication_keys": sorted(older_keys),
    }


def _span(result: RetrievedChunk, keys: dict[UUID, str]) -> RetrievedSpan:
    return RetrievedSpan(keys.get(result.document_id, "?"), result.char_start, result.char_end)


def _lineage_row(
    question_id: str, spans: Sequence[Any], ranked: Sequence[RetrievedSpan], older: set[str]
) -> dict[str, Any]:
    older_ranks = [rank for rank, r in enumerate(ranked, 1) if r.document in older]
    relevant_ranks = [
        rank
        for rank, r in enumerate(ranked, 1)
        if any(overlap_fraction(s, r) >= COVER_FRACTION for s in spans)
    ]
    first_older = min(older_ranks, default=None)
    first_relevant = min(relevant_ranks, default=None)
    return {
        "id": question_id,
        "older_documents": sorted(older),
        "first_older_rank": first_older,
        "first_relevant_rank": first_relevant,
        "older_in_top5": first_older is not None and first_older <= 5,
        "older_outranks_relevant": first_older is not None
        and (first_relevant is None or first_older < first_relevant),
    }


def write_report(report: dict[str, Any], out: Path) -> Path:
    folder: Path = out / str(report["run_id"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (folder / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return folder


def render_markdown(report: dict[str, Any]) -> str:
    env, data, names = report["environment"], report["dataset"], list(report["results"])
    lines = [
        f"# Retrieval comparison `{report['run_id']}`",
        "",
        f"- Dataset: `{data['name']}` v{data['version']}, {data['questions']} questions "
        f"(questions sha256 `{data['questions_sha256'][:12]}`, corpus manifest "
        f"`{data['corpus_manifest_sha256'][:12]}`)",
        f"- Embedding: `{env['embedding_model']}` @ `{env['embedding_revision'][:12]}`; "
        f"index fingerprint `{env['index_fingerprint'][:16]}`; BM25 {json.dumps(env['sparse'])}",
        f"- Reranker: {json.dumps(report['reranker'])}",
        f"- Search settings: {json.dumps(report['search_settings'])} (none tuned on this dataset)",
        f"- Environment: Python {env['python']}, {env['platform']}, CPU only: {env['cpu_only']}, "
        f"Qdrant {env['qdrant_server']}",
        "",
        "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Hit@5 |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        m = report["results"][name]["metrics"]

        def cell(key: str, m: dict[str, Any] = m) -> str:
            return f"{m[key]['mean']:.3f} [{m[key]['ci_low']:.2f}, {m[key]['ci_high']:.2f}]"

        lines.append(
            f"| {name} | {cell('recall@5')} | {cell('recall@10')} | {cell('reciprocal_rank@10')} "
            f"| {cell('ndcg@10')} | {cell('hit@5')} |"
        )
    lines += ["", "95% percentile bootstrap intervals over questions (1,000 resamples).", ""]
    if report["comparisons"]:
        lines += [
            "Paired per-question differences (bootstrap CI over questions):",
            "",
            "| Comparison | ΔRecall@5 | ΔRecall@10 | ΔMRR@10 | better / worse / same (R@5) |",
            "|---|---|---|---|---|",
        ]
        for name, deltas in report["comparisons"].items():
            cells = [
                f"{deltas[m]['mean']:+.3f} [{deltas[m]['ci_low']:+.2f}, "
                f"{deltas[m]['ci_high']:+.2f}]"
                for m in ("recall@5", "recall@10", "rr@10")
            ]
            r5 = deltas["recall@5"]
            lines.append(
                f"| {name} | "
                + " | ".join(cells)
                + f" | {r5['better']} / {r5['worse']} / {r5['same']} |"
            )
        lines.append("")
    categories = list(report["results"][names[0]]["by_category"])
    lines += [
        "Recall@5 by category:",
        "",
        "| Category | n | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for category in categories:
        n = report["results"][names[0]]["by_category"][category]["n"]
        values = [
            f"{report['results'][name]['by_category'][category]['recall@5']:.3f}" for name in names
        ]
        lines.append(f"| {category} | {n} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Older publications and table-of-contents results:",
        "",
        "| Configuration | older pub. in top 5 | older outranks first relevant "
        "| questions with TOC in top 5 | TOC share of top-5 results |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        older = report["results"][name]["older_publications"]
        toc = report["results"][name]["table_of_contents"]
        lines.append(
            f"| {name} | {older['with_older_in_top5']} / {older['questions']} "
            f"| {older['older_outranks_first_relevant']} / {older['questions']} "
            f"| {toc['questions_with_toc_in_top5']} / {data['questions']} "
            f"| {toc['share_of_top5_results']:.3f} |"
        )
    lines += [
        "",
        "| Latency p50 / p95 (ms) | total | embed | dense | sparse | rerank |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        lat = report["results"][name]["latency_ms"]

        def stage(key: str, lat: dict[str, Any] = lat) -> str:
            return f"{lat[key]['p50']} / {lat[key]['p95']}" if key in lat else "-"

        lines.append(
            f"| {name} | "
            + " | ".join(stage(k) for k in ("total", "embed", "dense", "sparse", "rerank"))
            + " |"
        )
    isolation = report["isolation"]
    lines += [
        "",
        f"Cross-tenant check: {isolation['foreign_results']} results from "
        f"{isolation['foreign_documents']} foreign-tenant documents "
        f"(checked: {isolation['foreign_org_checked']}). Model load seconds (excluded): "
        f"{json.dumps(report['load_seconds'])}.",
        "",
    ]
    return "\n".join(lines)
