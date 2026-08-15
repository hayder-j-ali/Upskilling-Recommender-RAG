"""Streamlit demo for the upskilling recommender.

A thin UI over the existing pipeline. The interesting code lives in
`learning_rec.recommender` and `learning_rec.retrieval`; this file just
wires those to widgets.

Run:
    streamlit run app/streamlit_app.py

Prereqs:
- `OPENAI_API_KEY` in environment (or .env) — needed for the dense and
  hybrid retrievers and for the LLM rerank stage. The BM25 retrieval-only
  path works without one.
- `python scripts/build_index.py --reset` must have been run at least once
  (the dense and hybrid retrievers load the FAISS index from disk).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning_rec.config import (
    CONTENT_FILE,
    EMPLOYEES_FILE,
    INDEX_DIR,
    NUM_RECOMMENDATIONS,
    TOP_K,
)
from learning_rec.recommender import rerank_with_llm, retrieve
from learning_rec.retrieval import build_retriever
from learning_rec.retrieval.factory import RetrieverKind

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Upskilling Recommender",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_data
def load_employees(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def get_retriever(kind: RetrieverKind):
    """Retrievers are heavy to build (FAISS load, BM25 indexing); cache them."""
    return build_retriever(kind, index_dir=INDEX_DIR, content_path=CONTENT_FILE)


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _needs_openai(kind: RetrieverKind, use_rerank: bool) -> bool:
    return kind != "bm25" or use_rerank


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

employees = load_employees(EMPLOYEES_FILE)
emp_options = {
    f"{row.employee_id} — {row['name']} ({row.role})": row.employee_id
    for _, row in employees.iterrows()
}
emp_label = st.sidebar.selectbox("Employee", options=list(emp_options.keys()))
emp_id = emp_options[emp_label]
emp = employees[employees.employee_id == emp_id].iloc[0]

retriever_kind: RetrieverKind = st.sidebar.radio(
    "Retrieval strategy",
    options=["dense", "bm25", "hybrid"],
    index=2 if _has_openai_key() else 1,
    help=(
        "**dense** — FAISS over OpenAI embeddings (the thesis baseline).  \n"
        "**bm25** — sparse lexical scoring, no API calls.  \n"
        "**hybrid** — Reciprocal Rank Fusion of dense + BM25 (recommended)."
    ),
)

use_rerank = st.sidebar.checkbox(
    "LLM re-rank with reasons",
    value=_has_openai_key(),
    help=(
        "Calls gpt-4o-mini to re-rank and explain each pick. Adds a few "
        "cents per click. Uncheck to see raw retrieval candidates only "
        "(no chat-model calls)."
    ),
)

top_k = st.sidebar.slider("Top-K candidates to retrieve", 5, 20, TOP_K)
n_recs = (
    st.sidebar.slider("Recommendations after rerank", 3, 10, NUM_RECOMMENDATIONS)
    if use_rerank
    else top_k
)

# Cost / prereq hint
if _needs_openai(retriever_kind, use_rerank):
    if not _has_openai_key():
        st.sidebar.error(
            "OPENAI_API_KEY not set. Either set it in your environment / "
            ".env, or pick **bm25** + uncheck **LLM re-rank** for a fully "
            "offline demo."
        )
else:
    st.sidebar.success("Running fully offline — no API calls on this click.")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("📚 Upskilling Recommender")
_README_URL = (
    "https://github.com/hayder-j-ali/Upskilling-Recommender-RAG#how-it-works"
)
st.markdown(
    "*Match employees to upskilling content with semantic search and an "
    f"LLM re-ranker. [See the README]({_README_URL}) for the architecture.*"
)

profile_col, results_col = st.columns([1, 2])

with profile_col:
    st.subheader("Employee profile")
    st.markdown(f"**Role** — {emp.role}")
    st.markdown(f"**Skills** — {emp.skills.replace(';', ', ')}")
    st.markdown(f"**Job** — {emp.job_description}")
    st.markdown(f"**Strengths** — {emp.strengths.replace(';', ', ')}")
    st.markdown(f"**Interests** — {emp.interests}")
    st.markdown(f"**Last course taken** — *{emp.last_course}*")

with results_col:
    st.subheader("Recommendations")

    run = st.button("Generate recommendations", type="primary")
    if not run:
        st.info(
            "Pick an employee and settings on the left, then click **Generate "
            "recommendations**. Try the same employee with different retrieval "
            "strategies to see how the picks change."
        )
    elif _needs_openai(retriever_kind, use_rerank) and not _has_openai_key():
        st.error(
            "This configuration needs an OpenAI API key. Set `OPENAI_API_KEY` "
            "in your environment or `.env`, or switch to **bm25** and uncheck "
            "**LLM re-rank** for a fully offline demo."
        )
    else:
        spinner_msg = (
            f"Retrieving ({retriever_kind})"
            + (" and re-ranking with the LLM…" if use_rerank else "…")
        )
        with st.spinner(spinner_msg):
            try:
                retriever = get_retriever(retriever_kind)
                candidates = retrieve(emp, retriever, k=top_k)
            except FileNotFoundError as e:
                st.error(
                    f"Could not load the index ({e}). Run "
                    "`python scripts/build_index.py --reset` first."
                )
                st.stop()

            if use_rerank:
                results = rerank_with_llm(emp, candidates, n=n_recs)
                for i, r in enumerate(results, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**{i}. {r['content_name']}** "
                            f"`{r['content_id']}`"
                        )
                        st.markdown(f"_{r['reason']}_")
            else:
                st.caption(
                    f"Showing top-{top_k} raw retrieval candidates "
                    f"(retriever score; no LLM rerank)."
                )
                for i, c in enumerate(candidates, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**{i}. {c['content_name']}** "
                            f"`{c['content_id']}` — score: {c['score']:.3f}"
                        )
                        st.caption(c["description"][:200] + "…")
