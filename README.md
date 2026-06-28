# Upskilling Recommender RAG

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

```
                 ┌──────────────────────────────────────────────┐
                 │       learning_content.csv (catalogue)       │
                 │  id, name, description, keywords, skills...  │
                 └──────────────────────┬───────────────────────┘
                                        │ ingest.py
                                        ▼
                          ┌─────────────────────────┐
                          │   OpenAI Embeddings     │
                          │ (text-embedding-3-small)│
                          └────────────┬────────────┘
                                       ▼
                              ┌─────────────────┐
                              │  FAISS  index   │
                              └────────┬────────┘
                                       │ similarity search (k=10)
       employee profile  ─── query ────┤
       (skills 2x weighted)            ▼
                                ┌─────────────┐
                                │ candidates  │
                                └──────┬──────┘
                                       │ profile + candidates
                                       ▼
                               ┌──────────────┐
                               │   LLM        │
                               │  re-ranker   │     →  top-5 JSON with reasons
                               │ (gpt-4o-mini)│
                               └──────────────┘
```

Two design choices worth calling out:

1. **Skill-weighted query.** When constructing the embedding query for an
   employee, the `skills` field is duplicated. In the thesis evaluation this
   small change reliably nudged the retriever toward skill-relevant content
   without needing a separate rerank stage on top of pure cosine similarity.
2. **LLM as re-ranker, not retriever.** The vector store retrieves a
   shortlist of `k=10`; the LLM only re-ranks and explains. This keeps
   inference cost bounded by `k`, not by catalogue size, and lets the LLM
   give human-readable justifications that an embedding score can't.

## Quick start

```bash
git clone https://github.com/hayderalijaan/Upskilling-Recommender-RAG.git
cd Upskilling-Recommender-RAG

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env to add your OPENAI_API_KEY

# (re)generate synthetic data — already committed, but you can regenerate
python scripts/generate_synthetic_data.py

# build the FAISS index from the catalogue
python scripts/build_index.py --reset

# produce recommendations for every employee
python scripts/recommend.py

# or, smoke-test on just one
python scripts/recommend.py --limit 1
```

Outputs land in `output/recommend_<employee_id>.json`.

## Configuration

All configuration is environment-driven via `.env`:

| Variable              | Default                    | Purpose                            |
| --------------------- | -------------------------- | ---------------------------------- |
| `OPENAI_API_KEY`      | _(required)_               | OpenAI API access                  |
| `CHAT_MODEL`          | `gpt-4o-mini`              | Re-ranker model                    |
| `EMBEDDING_MODEL`     | `text-embedding-3-small`   | Embedding model                    |
| `TEMPERATURE`         | `0.2`                      | Re-ranker sampling temperature     |
| `TOP_K`               | `10`                       | Candidates retrieved before rerank |
| `NUM_RECOMMENDATIONS` | `5`                        | Final results returned per user    |
| `DATA_DIR`            | `./data`                   | Where input CSVs live              |
| `INDEX_DIR`           | `./vector_store`           | Where the FAISS index is written   |
| `OUTPUT_DIR`          | `./output`                 | Where recommendations are written  |

## Project structure

```
Upskilling-Recommender-RAG/
├── data/                          # synthetic input data (committed)
│   ├── employees.csv
│   └── learning_content.csv
├── src/learning_rec/
│   ├── config.py                  # env-driven settings
│   ├── ingest.py                  # catalogue -> FAISS
│   ├── recommender.py             # employee -> top-5 with reasons
│   └── prompts.py                 # LLM system prompt
├── scripts/
│   ├── generate_synthetic_data.py # rebuild the demo dataset
│   ├── build_index.py             # CLI for ingest
│   └── recommend.py               # CLI for batch recommendations
├── output/                        # recommendation JSON (gitignored)
├── vector_store/                  # FAISS artifacts (gitignored)
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

## Roadmap

Planned extensions (not yet implemented in this public version):

- **Evaluation harness** — offline metrics (recall@k, MRR, LLM-as-judge
  relevance) over a held-out set, so README claims are backed by numbers.
- **Hybrid retrieval** — BM25 + dense ensemble for queries dominated by rare
  technical terms (e.g. specific tool names).
- **Streamlit UI** — one-page demo so reviewers can click through profiles
  without touching a terminal.

## Origin

This project is an anonymized reimplementation of the practical component of
my Master's thesis. The original was developed against a corporate Learning
Experience Platform; this version replaces all proprietary data, paths, and
naming with synthetic equivalents. The recommender logic, prompt design, and
evaluation methodology are unchanged from the thesis.

## License

MIT — see [LICENSE](LICENSE).
