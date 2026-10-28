"""Calibrate retrieval-only sufficiency (tuning set) or evaluate grounded answers (held-out set).

    python scripts/evaluate_answers.py calibrate --org <org> --kb <nist_kb> --index <index>
    python scripts/evaluate_answers.py evaluate --org <org> --kb <nist_kb> --injection-kb <kb> \
        --index <index> [--policies default,include] [--apply-authority]

``calibrate`` uses only the tuning questions (nist-security retrieval questions plus
nist-answers/v1/tuning.jsonl) and calls no model. ``evaluate`` uses only
nist-answers/v1/questions.jsonl. Reports are written to evals/reports/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from kip.answer_evaluation import (
    apply_authority,
    calibrate_sufficiency,
    documents_by_sha,
    load_items,
    run_answer_evaluation,
    sha256_file,
    tuning_signals,
)
from kip.answering import prepare_evidence
from kip.db.session import make_engine, session_factory
from kip.llm import answer_settings, make_provider
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.vectors import EmbedderCache, QdrantIndexFactory, RerankerCache

from rag.answering import AuthorityPolicy, SufficiencyThresholds

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "evals" / "datasets" / "nist-answers" / "v1"
RETRIEVAL = ROOT / "evals" / "datasets" / "nist-security" / "v1"
POLICIES = {
    "default": "current=include,unknown=include,superseded=demote,withdrawn=demote",
    "include": "current=include,unknown=include,superseded=include,withdrawn=include",
    "exclude": "current=include,unknown=include,superseded=exclude,withdrawn=exclude",
}


def corpus_shas() -> dict[str, str]:
    manifest = json.loads((RETRIEVAL / "corpus.json").read_text(encoding="utf-8"))
    shas = {doc["key"]: doc["sha256"] for doc in manifest["documents"]}
    shas["injection-note"] = sha256_file(ANSWERS / "injection-note.md")
    return shas


def write(report: dict[str, Any], out: Path, kind: str) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{kind}"
    folder = out / run_id
    folder.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, **report}
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["calibrate", "evaluate"])
    parser.add_argument("--org", type=UUID, required=True)
    parser.add_argument("--kb", type=UUID, required=True, help="NIST knowledge base.")
    parser.add_argument("--injection-kb", type=UUID, help="KB holding injection-note.md.")
    parser.add_argument("--index", type=UUID)
    parser.add_argument("--policies", default="default,include")
    parser.add_argument("--apply-authority", action="store_true")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "reports")
    args = parser.parse_args(argv)

    settings = Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    sessions = session_factory(engine)
    embedders, vectors, rerankers = (
        EmbedderCache(settings),
        QdrantIndexFactory(settings),
        RerankerCache(settings),
    )
    config = answer_settings(settings, rerank=False if args.no_rerank else None)
    try:
        if args.command == "calibrate":
            questions = [
                (json.loads(line)["question"], True)
                for line in (RETRIEVAL / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ] + [
                (json.loads(line)["question"], bool(json.loads(line)["answerable"]))
                for line in (ANSWERS / "tuning.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # Thresholds of 0 so every question yields its raw signals.
            raw = replace(config, thresholds=SufficiencyThresholds(min_coverage=0.0))
            signals = tuning_signals(
                questions,
                lambda q: prepare_evidence(
                    sessions,
                    embedders,
                    vectors,
                    org_id=args.org,
                    kb_ids=[args.kb],
                    question=q,
                    config=raw,
                    reranker_factory=rerankers,
                    index_version_id=args.index,
                ),
            )
            report: dict[str, Any] = {
                "kind": "sufficiency-calibration",
                "tuning_questions": len(questions),
                "settings": {
                    "mode": config.mode,
                    "rerank": config.rerank,
                    "top_k": config.top_k,
                    "budget_tokens": config.budget_tokens,
                },
                "result": calibrate_sufficiency(signals),
                "signals": [
                    {"answerable": a, "coverage": c, "top_rerank": t} for a, c, t in signals
                ],
            }
            folder = write(report, args.out, "sufficiency-calibration")
            print(json.dumps({"report": str(folder), **report["result"]}, indent=2))
            return 0

        versions = documents_by_sha(sessions, args.org, corpus_shas())
        if args.apply_authority:
            authority = json.loads((ANSWERS / "authority.json").read_text(encoding="utf-8"))
            apply_authority(sessions, args.org, versions, authority)
        kbs = {"nist": [args.kb], "nist+injection": [args.kb, args.injection_kb or args.kb]}
        provider = make_provider(settings)
        items = load_items(ANSWERS / "questions.jsonl")
        evaluated = run_answer_evaluation(
            sessions=sessions,
            store=LocalObjectStore(settings.storage_root),
            embedder_factory=embedders,
            vector_index_factory=vectors,
            reranker_factory=rerankers,
            provider=provider,
            items=items,
            org_id=args.org,
            kbs=kbs,
            versions=versions,
            settings=config,
            policies={
                name: AuthorityPolicy.parse(POLICIES[name]) for name in args.policies.split(",")
            },
            index_version_id=args.index,
        )
        report = {
            "kind": "answer-evaluation",
            "dataset": {
                "questions_sha256": sha256_file(ANSWERS / "questions.jsonl"),
                "authority_sha256": sha256_file(ANSWERS / "authority.json"),
                "items": len(items),
            },
            **evaluated,
        }
        folder = write(report, args.out, "answers")
        print(
            json.dumps(
                {
                    "report": str(folder),
                    "summary": {k: v["summary"] for k, v in report["results"].items()},
                },
                indent=2,
                default=str,
            )
        )
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
