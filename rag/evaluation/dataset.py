"""Evaluation dataset: questions with quote labels, resolved to canonical-text spans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.evaluation.relevance import Span, locate_quote, locate_quote_all

CATEGORIES = {"factual", "paraphrase", "identifier", "version_sensitive", "multi_document"}


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    labels: list[dict[str, Any]]
    outdated: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    questions: list[Question]
    manifest: dict[str, Any]
    questions_sha256: str
    manifest_sha256: str

    @property
    def document_keys(self) -> set[str]:
        return {document["key"] for document in self.manifest["documents"]}


def load_dataset(folder: Path) -> Dataset:
    manifest_bytes = (folder / "corpus.json").read_bytes()
    question_bytes = (folder / "questions.jsonl").read_bytes()
    manifest = json.loads(manifest_bytes)
    keys = {document["key"] for document in manifest["documents"]}
    questions: list[Question] = []
    seen: set[str] = set()
    for line_number, line in enumerate(question_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        question = Question(
            raw["id"], raw["category"], raw["question"], raw["relevant"], raw.get("outdated", [])
        )
        problems = []
        if question.id in seen:
            problems.append("duplicate id")
        if question.category not in CATEGORIES:
            problems.append(f"unknown category {question.category!r}")
        if not question.labels:
            problems.append("no relevant labels")
        for label in question.labels:
            if label["doc"] not in keys or label.get("grade") not in (1, 2):
                problems.append(f"bad label {label.get('doc')!r}/{label.get('grade')!r}")
        if not set(question.outdated) <= keys:
            problems.append("unknown outdated document key")
        if problems:
            raise ValueError(f"questions.jsonl line {line_number}: {'; '.join(problems)}")
        seen.add(question.id)
        questions.append(question)
    return Dataset(
        manifest["name"],
        manifest["version"],
        questions,
        manifest,
        hashlib.sha256(question_bytes).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def resolve_spans(question: Question, canonical: Mapping[str, str]) -> list[Span]:
    """Locate every label in its document's canonical text (raises if missing/ambiguous)."""
    spans: list[Span] = []
    for label in question.labels:
        text = canonical[label["doc"]]
        if label.get("all_occurrences"):
            locations = locate_quote_all(label["quote"], text)
        else:
            locations = [locate_quote(label["quote"], text)]
        spans.extend(Span(label["doc"], start, end, label["grade"]) for start, end in locations)
    return spans
