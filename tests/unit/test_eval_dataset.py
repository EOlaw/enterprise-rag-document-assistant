from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.evaluation.dataset import Question, load_dataset, resolve_spans
from rag.evaluation.relevance import QuoteNotFoundError

ROOT = Path(__file__).resolve().parents[2]
NIST = ROOT / "evals" / "datasets" / "nist-security" / "v1"


def test_committed_dataset_is_well_formed() -> None:
    dataset = load_dataset(NIST)
    assert len(dataset.questions) >= 30
    assert dataset.document_keys == {
        "sp800-61r2",
        "sp800-61r3",
        "csf-1.1",
        "csf-2.0",
        "sp800-63b-4",
    }
    for document in dataset.manifest["documents"]:
        assert len(document["sha256"]) == 64 and document["url"].startswith("https://")
    for question in dataset.questions:
        for label in question.labels:
            assert len(label["quote"].split()) >= 3, (
                f"{question.id}: quote too short to be specific"
            )


def write(folder: Path, questions: list[dict]) -> Path:
    manifest = {"name": "t", "version": "1", "documents": [{"key": "doc"}]}
    (folder / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions), encoding="utf-8"
    )
    return folder


def question(**overrides: object) -> dict:
    base = {"id": "q1", "category": "factual", "question": "?",
            "relevant": [{"doc": "doc", "quote": "alpha beta", "grade": 2}]}  # fmt: skip
    return {**base, **overrides}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (question(category="vibes"), "unknown category"),
        (question(relevant=[]), "no relevant labels"),
        (question(relevant=[{"doc": "missing", "quote": "x y z", "grade": 2}]), "bad label"),
        (question(relevant=[{"doc": "doc", "quote": "x y z", "grade": 3}]), "bad label"),
        (question(outdated=["missing"]), "outdated"),
    ],
)
def test_malformed_questions_are_rejected(tmp_path: Path, bad: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_dataset(write(tmp_path, [bad]))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(write(tmp_path, [question(), question()]))


def test_resolve_spans_single_and_all_occurrences() -> None:
    text = "alpha beta gamma. alpha beta delta."
    single = Question("q", "factual", "?", [{"doc": "doc", "quote": "beta gamma", "grade": 2}])
    assert [(s.start, s.end) for s in resolve_spans(single, {"doc": text})] == [(6, 16)]
    repeated = Question("q", "factual", "?", [{"doc": "doc", "quote": "alpha beta", "grade": 1,
                                                "all_occurrences": True}])  # fmt: skip
    assert len(resolve_spans(repeated, {"doc": text})) == 2
    ambiguous = Question("q", "factual", "?", [{"doc": "doc", "quote": "alpha beta", "grade": 2}])
    with pytest.raises(QuoteNotFoundError, match="ambiguous"):
        resolve_spans(ambiguous, {"doc": text})
