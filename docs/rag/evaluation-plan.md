# Evaluation Plan

No retrieval or generation change is accepted on anecdotes. Every reported number
records: dataset version + hash, index fingerprint (parser, chunking, embedding model
revision), retrieval config, LLM + prompt version, hardware, and date.

## 1. Gold dataset (`evals/datasets/<name>/<version>/`)

**Relevance is labelled on source spans, not chunk IDs.** Each relevant passage is
`(document_key, page, quote)` resolved to canonical-text offsets. A chunk is relevant
(grade 2) if it contains ≥ 50% of a labelled span, partially relevant (grade 1) if it
overlaps it. This is what makes different chunking strategies comparable on the same
labels.

```jsonc
{
  "id": "sec-0042",
  "question": "How long must incident records be retained?",
  "kb_scope": ["security-compliance"],
  "category": "outdated_version",        // see quotas below
  "answerable": true,
  "relevant_spans": [{ "doc": "nist-sp-800-61r3", "page": 12, "quote": "...", "grade": 2 }],
  "reference_facts": ["..."],             // atomic facts a correct answer must state
  "must_not_cite": ["nist-sp-800-61r2"],  // superseded / unauthorized sources
  "acting_user": "reader@tenant-a"        // authorization context for the query
}
```

Target size ≥ 250 questions, split **dev (40%) / test (60%)**; all tuning uses dev only.

| Category | Quota | Purpose |
|---|---|---|
| Simple factual | 50 | baseline |
| Paraphrased (no lexical overlap with source) | 40 | dense strength |
| Exact identifier | 40 | sparse strength, identifier preservation |
| Multi-document synthesis | 25 | recall across docs, context selection |
| Conflicting / outdated version | 25 | version metadata, conflict disclosure |
| Unanswerable (in-domain, not in corpus) | 30 | abstention |
| Unauthorized (answer exists only in another tenant/restricted doc) | 20 | must abstain *and* retrieve nothing unauthorized |
| Prompt injection (payload document in scope) | 20 | must not follow embedded instructions |

Ground truth is written and reviewed by humans (two annotators on a 20% overlap sample;
Cohen's κ reported). An LLM may *draft* candidate questions; it never labels its own
answers as correct.

## 2. Retrieval metrics

Recall@k, Precision@k, Hit-Rate@k, MRR, nDCG@k (graded), for k ∈ {1, 3, 5, 10}, plus
context tokens, index size and p50/p95 retrieval latency. Reported with **95% bootstrap
confidence intervals** (1,000 resamples over questions). Two configurations are compared
with a **paired bootstrap / permutation test** on per-question scores; a change is called
an improvement only if the CI of the paired difference excludes 0.

## 3. Generation metrics

| Metric | Definition | Measurement |
|---|---|---|
| Answer correctness | fraction of `reference_facts` stated correctly, no contradicted fact | human on test sample; LLM judge validated vs human (κ reported) |
| Groundedness / unsupported-claim rate | claims not supported by cited evidence ÷ claims | claim split + NLI/LLM judge; human audit sample |
| Citation support | cited passage supports the citing sentence | as above |
| Citation validity | cited label maps to provided evidence, accessible, correct version | deterministic, 100% required |
| Abstention quality | precision/recall of abstaining on unanswerable + unauthorized items | deterministic |
| Answer relevance | answer addresses the question | judge + human sample |

Judge limitations (position bias, verbosity bias, self-preference) are documented; the
judge model differs from the generator where feasible.

## 4. Security invariants (pass/fail, must be 100%)

* zero chunks from another organization retrieved, for any query, ever;
* zero restricted chunks retrieved for users without a grant;
* zero citations to inaccessible or superseded-and-hidden versions;
* injection items: answer contains no evidence of followed instructions (pattern checks +
  human review of failures).

## 5. System metrics

Per-stage latency (analysis, embedding, dense, sparse, fusion, rerank, generation first
token, total), tokens in/out, cost per query when the provider prices are known, error
rate, throughput under a stated concurrency. Hardware is always reported; CPU-only numbers
are labelled as such.

## 6. Experiment matrix (each run produces `evals/reports/<run_id>/report.json` + `.md`)

1. Dense only · 2. BM25 only · 3. Hybrid (RRF) · 4. Hybrid + rerank
5. Chunk sizes 128/200/254, overlap 0/32/64, page boundaries on/off, structure-aware vs fixed
6. Top-k 3/5/10 into context · 7. Context budget 1k/2k/4k tokens
8. Embedding models: MiniLM vs bge-small vs e5-small (if 1–7 show retrieval is the bottleneck)

Baseline = config 1 with `fixed_tokens` chunking. Every later claim of improvement is
relative to a recorded baseline run on the same dataset version.

## 7. Regression gate (CI, Phase 9)

A small frozen **smoke set** (~40 dev questions, synthetic + public docs, runs on CPU in
minutes) runs on every PR touching `rag/`. The PR fails if Recall@5 or citation validity
drop beyond the bootstrap noise band of the last accepted run, or any security invariant
fails.
