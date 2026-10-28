"""Evaluate retrieval configurations end to end (Qdrant -> PostgreSQL re-check).

    python scripts/evaluate_retrieval.py evals/datasets/nist-security/v1 \
        --org <org_id> --kb <kb_id> [--index <index_version_id>] [--foreign-org <org_id>] \
        [--configs dense,bm25,hybrid,hybrid+rerank]

The corpus must already be ingested and published in that organization/KB (see
evals/datasets/nist-security/README.md). Documents are matched to dataset keys by the
pinned SHA-256 in corpus.json, and labels are resolved against the *stored* canonical
text of the published version, so a parser change that moves text fails loudly.

Writes evals/reports/<run_id>/report.json and report.md. With ~40 questions the bootstrap
intervals are wide; small differences between configurations are not meaningful. Exit
status is 1 if any result belongs to the foreign tenant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

import httpx
from kip.db.session import make_engine, session_factory
from kip.evaluation import (
    DEFAULT_CONFIGS,
    RunConfig,
    SearchSettings,
    run_evaluation,
    write_report,
)
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import EmbedderCache, QdrantIndexFactory, RerankerCache

from rag.evaluation.dataset import load_dataset

CONFIGS = {config.name: config for config in DEFAULT_CONFIGS}


def parse_configs(value: str) -> list[RunConfig]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"choose from {sorted(CONFIGS)}; got {unknown or value}")
    return [CONFIGS[name] for name in names]


def qdrant_version(url: str) -> str | None:
    try:
        return str(httpx.get(url, timeout=5).json().get("version"))
    except (httpx.HTTPError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--org", type=UUID, required=True)
    parser.add_argument("--kb", type=UUID, required=True)
    parser.add_argument("--index", type=UUID)
    parser.add_argument(
        "--foreign-org", type=UUID, help="Tenant whose documents must never appear."
    )
    parser.add_argument(
        "--configs",
        type=parse_configs,
        default=list(DEFAULT_CONFIGS),
        help="Comma-separated; the first is the baseline for paired differences.",
    )
    parser.add_argument("--out", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    settings = Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    try:
        report = run_evaluation(
            sessions=session_factory(engine),
            store=LocalObjectStore(settings.storage_root),
            embedder_factory=EmbedderCache(settings),
            vector_index_factory=QdrantIndexFactory(settings),
            reranker_factory=RerankerCache(settings),
            dataset=load_dataset(args.dataset),
            org_id=args.org,
            kb_id=args.kb,
            index_version_id=args.index,
            foreign_org=args.foreign_org,
            configs=args.configs,
            search_settings=SearchSettings(
                overfetch=settings.search_overfetch,
                max_candidates=settings.search_max_candidates,
                hybrid_candidates=settings.hybrid_candidates,
                rrf_k=settings.rrf_k,
                rerank_candidates=settings.rerank_candidates,
            ),
            qdrant_server=qdrant_version(settings.qdrant_url),
            device=settings.embed_device,
        )
    finally:
        engine.dispose()
    folder = write_report(report, args.out)
    summary = {
        name: {
            "recall@5": round(result["metrics"]["recall@5"]["mean"], 3),
            "mrr@10": round(result["metrics"]["reciprocal_rank@10"]["mean"], 3),
        }
        for name, result in report["results"].items()
    }
    print(json.dumps({"report": str(folder), "summary": summary}, indent=2))
    return 1 if report["isolation"]["foreign_results"] else 0


if __name__ == "__main__":
    sys.exit(main())
