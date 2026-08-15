# Upskilling Recommender RAG

[![CI](https://github.com/hayder-j-ali/Upskilling-Recommender-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/hayder-j-ali/Upskilling-Recommender-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Match employees to upskilling content with semantic search and an LLM re-ranker.

A retrieval-augmented generation pipeline that recommends learning content
(courses, videos, articles) to employees based on their skills, job description,
strengths, and interests. Built originally as the practical component of my
Master's thesis at a large industrial corporation; this public version uses
fully synthetic data and is structured for portfolio review.

---

## What it does

Given an employee profile like this:

```
role:            Data Engineer
skills:          Python, SQL, Airflow, ETL
job_description: Build and maintain batch and streaming pipelines feeding the
                 analytics warehouse.
strengths:       Reliability mindset, Systems thinking, Mentoring
interests:       Streaming systems, observability
```

it returns the top-5 most relevant learning items from the catalogue, each
with a short justification:

```json
[
  {
    "content_id": "C0004",
    "content_name": "Streaming Data with Kafka",
    "reason": "Directly supports the employee's interest in streaming systems and extends their ETL/Airflow skill set toward event-driven pipelines."
  },
  {
    "content_id": "C0010",
    "content_name": "Observability with OpenTelemetry",
    "reason": "Maps to the stated interest in observability and a Data Engineer's responsibility for pipeline reliability."
  }
]
```

## How it works

![Pipeline diagram: the content catalogue is ingested into two indexes in
parallel — dense FAISS vectors from Gemini embeddings, and sparse BM25.
Each returns its top 30, RRF fusion combines them into 10 candidates, and
an employee profile (with skills weighted twice) becomes the query. A
Gemini flash re-ranker turns those candidates into a top-5 JSON list with
a reason for each pick.](docs/infograph.png)


A `--retriever {dense,bm25,hybrid}` flag selects which retrieval strategy
runs upstream of the LLM; dense matches the original thesis baseline. Hybrid
is the sensible default for real catalogues, where queries mix rare tool
names with loose descriptive language — but note that on this repo's
synthetic benchmark **BM25 scores highest**, for reasons examined in
[Results](#results). The flag exists so that claim stays testable rather
than assumed.

Design choices worth calling out:

1. **Hybrid retrieval via RRF.** BM25 catches exact lexical matches for
   rare technical terms ("Kubernetes", "dbt", "OpenTelemetry") that dense
   embeddings tend to smooth over; dense catches semantic matches BM25
   misses ("orchestration" ↔ "Airflow"). [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
   combines their rankings robustly without needing score normalization.
2. **Skill-weighted query.** When constructing the query for an employee,
   the `skills` field is duplicated. In the thesis evaluation this small
   change reliably nudged the retriever toward skill-relevant content;
   for BM25 it boosts the term frequency of skill tokens, keeping signals
   aligned across both retrievers.
3. **LLM as re-ranker, not retriever.** Retrieval returns a shortlist of
   `k=10`; the LLM only re-ranks and explains. This keeps inference cost
   bounded by `k`, not by catalogue size, and lets the LLM give
   human-readable justifications that retrieval scores can't.

## Quick start

```bash
git clone https://github.com/hayder-j-ali/Upskilling-Recommender-RAG.git
cd Upskilling-Recommender-RAG

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env to add your GOOGLE_API_KEY (get one free at aistudio.google.com/apikey)

# (re)generate synthetic data — already committed, but you can regenerate
python scripts/generate_synthetic_data.py

# build the FAISS index from the catalogue
python scripts/build_index.py --reset

# produce recommendations for every employee (defaults to dense retrieval)
python scripts/recommend.py

# or use hybrid retrieval (BM25 + dense fused with RRF)
python scripts/recommend.py --retriever hybrid

# smoke-test on just one
python scripts/recommend.py --limit 1
```

Outputs land in `output/recommend_<employee_id>.json`.

## Configuration

All configuration is environment-driven via `.env`:

| Variable              | Default                    | Purpose                            |
| --------------------- | -------------------------- | ---------------------------------- |
| `GOOGLE_API_KEY`      | _(required)_               | Gemini API access                  |
| `CHAT_MODEL`          | `gemini-flash-latest`      | Re-ranker model                    |
| `EMBEDDING_MODEL`     | `models/gemini-embedding-2`| Embedding model                    |
| `TEMPERATURE`         | `0.2`                      | Re-ranker sampling temperature     |
| `TOP_K`               | `10`                       | Candidates retrieved before rerank |
| `NUM_RECOMMENDATIONS` | `5`                        | Final results returned per user    |
| `DATA_DIR`            | `./data`                   | Where input CSVs live              |
| `INDEX_DIR`           | `./vector_store`           | Where the FAISS index is written   |
| `OUTPUT_DIR`          | `./output`                 | Where recommendations are written  |

> **Changing `EMBEDDING_MODEL` requires rebuilding the index.** The build
> records which model produced the vectors, and the dense retriever refuses
> to load an index built by a different one:
>
> ```
> The index at ./vector_store was built with 'models/gemini-embedding-2'
> but EMBEDDING_MODEL is 'models/gemini-embedding-001'. …
> Rebuild it with:  python scripts/build_index.py --reset
> ```
>
> The check exists because this failure is otherwise silent. Every Gemini
> embedding model here emits 3072 dimensions, so mismatched vectors still
> compare cleanly and FAISS returns a confidently mis-ordered list rather
> than an error — the recommendations look plausible and are wrong. A
> dimension mismatch would at least crash.

## Project structure

```
Upskilling-Recommender-RAG/
├── data/                          # synthetic input data (committed)
│   ├── employees.csv
│   └── learning_content.csv
├── src/learning_rec/
│   ├── config.py                  # env-driven settings
│   ├── ingest.py                  # catalogue -> FAISS
│   ├── recommender.py             # retrieve(), rerank_with_llm(), recommend()
│   ├── prompts.py                 # LLM system prompts
│   ├── vector_store.py            # FAISS index: build, save/load, search
│   ├── index_meta.py              # records/verifies which model built the index
│   ├── llm_utils.py               # response parsing + API-error classification
│   ├── retrieval/
│   │   ├── base.py                # Retriever protocol + Candidate shape
│   │   ├── dense.py               # FAISS / Gemini-embeddings retriever
│   │   ├── bm25.py                # in-memory BM25 retriever
│   │   ├── hybrid.py              # RRF fusion
│   │   └── factory.py             # build_retriever("dense"|"bm25"|"hybrid")
│   └── eval/
│       ├── ground_truth.py        # rule-based relevance labels
│       ├── metrics.py             # recall@k, MRR, precision@k
│       ├── judge.py               # LLM-as-judge
│       └── pipeline.py            # orchestrates an eval run
├── app/
│   └── streamlit_app.py           # interactive demo UI
├── scripts/
│   ├── generate_synthetic_data.py # rebuild the demo dataset
│   ├── build_index.py             # CLI for ingest
│   ├── recommend.py               # CLI for batch recommendations
│   └── run_eval.py                # CLI for evaluation
├── output/                        # recommendation JSON (gitignored)
├── vector_store/                  # index.faiss + documents.json + index_meta.json (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Plugging in your own data

The pipeline doesn't know or care that the demo dataset is synthetic. Replace
the two CSVs with your own, keeping the columns:

- `learning_content.csv`: `content_id, content_name, content_description, content_language, duration_seconds, keywords, skills`
- `employees.csv`: `employee_id, name, role, skills, job_description, strengths, interests, last_course`

`keywords` and `skills` may be semicolon-separated, comma-separated, or
Python-literal lists — `ingest.to_list()` normalizes them.

## Evaluation

The eval harness measures the recommender at two stages independently:

- **Retrieval** — does the chosen retriever surface the relevant items at all?
  Metrics: `recall@10`, `precision@10`, `mrr@10`. Compare strategies with
  `--retriever {dense,bm25,hybrid}`.
- **End-to-end** — does the LLM rerank stage pick good items? Metric:
  `precision@5` against the ground-truth labels, plus an optional
  **LLM-as-judge** that rates each recommendation on a 4-point relevance scale.

**Ground truth** is rule-based: a course is "relevant" to an employee iff
their normalized skill sets share at least 2 elements (case-insensitive).
This is a *self-consistent benchmark* — it lets us compare retrieval methods
on the same yardstick, not an absolute relevance score. The threshold and
rationale are documented in [`src/learning_rec/eval/ground_truth.py`](src/learning_rec/eval/ground_truth.py).

```bash
# full eval: retrieval + rerank, no LLM judge
python scripts/run_eval.py

# cheap smoke run: 3 employees, retrieval only (skips chat-model calls)
python scripts/run_eval.py --limit 3 --no-rerank

# BM25 retrieval-only — fully offline, no Gemini calls at all
python scripts/run_eval.py --retriever bm25 --no-rerank

# compare hybrid retrieval (BM25 + dense) end-to-end
python scripts/run_eval.py --retriever hybrid

# add LLM-as-judge (extra Gemini calls)
python scripts/run_eval.py --judge
```

Outputs `output/eval_report.json` (full per-employee detail) and
`output/eval_summary.md` (paste-into-README summary).

## Results

Measured on the 25 synthetic employees against the 37-item catalogue, using
`gemini-embedding-2` and `gemini-flash-latest`. Reproduce with
`python scripts/run_eval.py --retriever {dense,bm25,hybrid} [--judge]`.

### Retrieval

Every relevance label comes from the skill-overlap rule described above, so
the threshold governs how many employees have labels at all. Both settings
are reported because the conclusion should not rest on one arbitrary cutoff:

**`MIN_SKILL_OVERLAP = 1`** — all 25 employees labeled (mean 5.2 relevant items):

| Retriever | recall@10 | MRR@10 | precision@10 |
| --------- | --------- | ------ | ------------ |
| BM25      | **0.880** | **0.960** | **0.408** |
| Dense     | 0.714     | 0.913  | 0.328        |
| Hybrid    | 0.839     | 0.950  | 0.376        |

**`MIN_SKILL_OVERLAP = 2`** (the default) — only 8 of 25 employees labeled,
scored over that subset:

| Retriever | recall@10 | MRR@10 | precision@10 |
| --------- | --------- | ------ | ------------ |
| BM25      | **1.000** | **0.938** | **0.125** |
| Dense     | 0.875     | 0.812  | 0.113        |
| Hybrid    | **1.000** | 0.844  | **0.125**    |

**BM25 wins on every metric at both thresholds — hybrid does not beat it.**

That is the opposite of what this project set out to show, and it is reported
as measured. The likely reason is a confound rather than a fact about
retrieval: relevance is *defined* as overlapping skill tokens, and matching
literal tokens is precisely what BM25 optimizes. The benchmark rewards the
lexical retriever by construction, so this result does not establish that
BM25 is the better choice in general — only that it wins the game these
labels describe. Hybrid does reliably recover the recall dense loses
(0.839 vs 0.714), which is the behavior RRF is meant to provide.

### End-to-end (hybrid retrieval + LLM re-ranking)

| Metric | Value |
| ------ | ----- |
| Rule-based precision@5 | 0.032 |
| LLM-as-judge mean score (0–1) | 0.810 |

Those two disagree sharply, and the disagreement is the most useful thing the
harness produced. Judge ratings across all 125 recommendations (25 employees
x top-5):

| Rating | Count | Share |
| ------ | ----- | ----- |
| highly relevant   | 60 | 48.0% |
| relevant          | 60 | 48.0% |
| somewhat relevant | 5  | 4.0%  |
| not relevant      | 0  | 0.0%  |

**96% rated relevant or better; none rated irrelevant.** Split by whether the
rule-based labels considered an employee to have any relevant content at all:

| Employee group | LLM judge | Rule-based precision@5 |
| -------------- | --------- | ---------------------- |
| Has labels (n=8)  | 0.780 | 0.100 |
| No labels (n=17)  | 0.824 | **0.000 — by construction** |

For 17 of 25 employees the rule finds no relevant course in the catalogue, so
their precision@5 can only ever be zero no matter what the system returns.
The judge, which never consults those labels, rates the same recommendations
0.824. The 0.032 figure is therefore measuring the sparsity of the labels
rather than the quality of the output.

### What to take from this

The rule-based labels are cheap, transparent, and auditable, which is why
they are here — but they are too sparse and too lexical to referee this
system on their own. The honest summary is that retrieval surfaces relevant
material reliably, the LLM re-ranker produces recommendations an independent
model judges relevant 96% of the time, and the rule-based precision figure
should not be read as a quality score. A stronger benchmark would need
human-curated labels, which is out of scope for a synthetic portfolio dataset.

> Re-running shifts these numbers slightly: the chat model is sampled at
> `temperature=0.2`, and `gemini-flash-latest` is an alias Google repoints at
> its current flash model.

## Demo UI

A Streamlit app wraps the same pipeline so you can click through profiles
without touching a terminal. Pick an employee, choose a retriever
(`dense` / `bm25` / `hybrid`), toggle LLM re-ranking on or off, get
recommendations.

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Opens at <http://localhost:8501>. The app adapts its defaults to your
environment: with no `GOOGLE_API_KEY` set it starts on **bm25** with
re-ranking off, so the demo runs fully offline with no API calls.

![Streamlit demo screenshot](docs/screenshot.png)

*Above: hybrid retrieval with LLM re-ranking on. Each card carries the
model's one-line justification, tied back to a specific skill, job
requirement, or interest from the profile on the left. Unchecking
**LLM re-rank** shows the raw retriever scores instead, and pairing that
with **bm25** runs the demo fully offline with no API calls.*

## Development

```bash
pip install -e ".[dev]"   # includes the UI extra so all tests run

# lint
ruff check .

# tests (offline only — no API calls)
pytest -v
```

CI runs both on every push and PR; see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Origin

This project is an anonymized reimplementation of the practical component of
my Master's thesis. The original was developed against a corporate Learning
Experience Platform; this version replaces all proprietary data, paths, and
naming with synthetic equivalents. The recommender logic, prompt design, and
evaluation methodology are unchanged from the thesis.

One deliberate deviation: the thesis used OpenAI's API (`text-embedding-3-small`
+ `gpt-3.5-turbo`). This public version runs on Gemini instead — its free tier
means anyone cloning the repo can run the full demo without setting up billing,
which matters more for a portfolio project meant to be tried, not just read.

## License

MIT — see [LICENSE](LICENSE).
