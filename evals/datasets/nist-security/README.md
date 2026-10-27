# `nist-security` retrieval dataset

A small, public, reproducible retrieval benchmark over five NIST publications.

## Corpus and provenance

[`v1/corpus.json`](v1/corpus.json) pins each document's URL, SHA-256 and byte size, plus
publication status and authorship notes. Documents are **not** stored in this repository.

| Key | Document | Status at retrieval (2026-10-27) |
|---|---|---|
| `sp800-61r2` | SP 800-61 Rev. 2, Computer Security Incident Handling Guide (2012) | Withdrawn 2025-04-03, superseded by Rev. 3 (notice on page 1 of the PDF) |
| `sp800-61r3` | SP 800-61r3, Incident Response Recommendations… (2025) | Current |
| `csf-1.1` | Cybersecurity Framework 1.1 (2018) | Earlier version of the CSF |
| `csf-2.0` | Cybersecurity Framework 2.0 (2024) | Current |
| `sp800-63b-4` | SP 800-63B-4, Authentication and Authenticator Management (2025) | Current |

**License basis (verified 2026-10-27).** NIST's statement for Technical Series
publications says works by NIST employees are not subject to US copyright, and that, to
the extent NIST asserts rights outside the US, it grants a worldwide royalty-free right to
reprint and create derivative works. It also says some works may be written by third
parties and be subject to copyright. Three documents list non-NIST co-authors; they are
recorded in the manifest. This repository redistributes none of the documents. The
dataset contains only short quotations used as relevance labels. This is a factual record,
not legal advice.

## Questions

[`v1/questions.jsonl`](v1/questions.jsonl) has 38 questions: 22 factual, 8 paraphrase,
3 identifier, 3 version-sensitive and 2 multi-document. Each label is a **quote** from a
document, located in the stored canonical text at evaluation time. A retrieved chunk
covers a label if it contains at least 50% of the quoted span, so the labels stay valid
across chunking configurations. Missing or ambiguous quotes fail the run instead of
attaching to the wrong passage. Labels marked `all_occurrences` accept any of a few
repeated, equivalent passages. `outdated` lists superseded documents that should ideally
not appear for that question.

**Authorship and review.** The questions and labels were written by an AI coding
assistant from the extracted document text. They have **not** been independently reviewed
by a human, and only one annotator was involved, so inter-annotator agreement cannot be
reported. Treat results as a development baseline. Review the labels before using them to
make claims, especially the paraphrase items, where the assistant chose the wording.

Not covered yet: unanswerable questions (abstention needs Phase 5), unauthorized-access
items beyond the cross-tenant check, and prompt-injection items.

## Reproduce

```powershell
python scripts/fetch_corpus.py evals/datasets/nist-security/v1/corpus.json   # verifies SHA-256
$rev = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
$a = kip bootstrap --slug eval-nist --name "NIST evaluation" --kb-name "NIST security" --revision $rev | ConvertFrom-Json
$b = kip bootstrap --slug eval-foreign --name "Foreign tenant" --kb-name "Foreign security" --revision $rev | ConvertFrom-Json
Get-ChildItem data/corpus/nist-security/*.pdf | ForEach-Object { kip submit $_.FullName --org $a.org_id --kb $a.kb_id --index $a.index_version_id }
# Same content in a second tenant, so isolation is tested with identical text:
foreach ($f in "NIST.CSWP.29.pdf", "NIST.SP.800-61r3.pdf") { kip submit "data/corpus/nist-security/$f" --org $b.org_id --kb $b.kb_id --index $b.index_version_id }
kip-worker --until-idle
python scripts/evaluate_retrieval.py evals/datasets/nist-security/v1 --org $a.org_id --kb $a.kb_id --foreign-org $b.org_id
```

The script exits non-zero if any result belongs to the foreign tenant.

## First result: dense retrieval baseline

Run [`20261027T100544Z-dense-120db434`](../../reports/20261027T100544Z-dense-120db434/report.md):
MiniLM-L6-v2 at `1110a243`, structure-aware chunks of 200 tokens with 32 tokens of overlap,
Qdrant 1.18.3, CPU only, Windows 11, Python 3.13.3.

| Metric (38 questions) | @1 | @5 | @10 |
|---|---|---|---|
| Recall | 0.25 [0.12, 0.39] | 0.55 [0.39, 0.70] | 0.71 [0.57, 0.86] |
| Hit rate | 0.26 | 0.58 | 0.74 |
| MRR | 0.26 | 0.37 | 0.39 |
| nDCG | 0.26 | 0.39 | 0.44 |

* By category (Recall@5): factual 0.68 (n=22), paraphrase 0.25 (n=8), identifier
  0.33 (n=3), version-sensitive 0.67 (n=3), multi-document 0.50 (n=2). The small
  categories are anecdotal.
* For 2 of 3 version-sensitive questions with a superseded source, the superseded
  document appeared in the top 5. Nothing ranks by recency or authority yet.
* Cross-tenant check: 0 results from the foreign tenant's 2 documents (identical content).
* Latency p50/p95: 46/62 ms end to end (embedding 8/10, Qdrant 24/40, PostgreSQL
  re-check 8/10). Model load (6.4 s) is excluded.
* A second run produced identical metrics and identical top-5 lists for every question.
  At 1,305 points Qdrant performs exact search (below its full-scan threshold).

Brackets are 95% percentile bootstrap intervals. With 38 questions they are wide, so only
large differences between future configurations will be distinguishable.
