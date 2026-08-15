"""Streamlit demo for the upskilling recommender.

A thin UI over the existing pipeline. The interesting code lives in
`learning_rec.recommender` and `learning_rec.retrieval`; this file just
wires those to widgets.

Run:
    streamlit run app/streamlit_app.py

Prereqs:
- `GOOGLE_API_KEY` in environment (or .env) — needed for the dense and
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
from google.genai.errors import APIError

# Reaches into a private module (`_common`) because langchain_google_genai
# doesn't re-export this at the package root. Necessary: GoogleGenerativeAIEmbeddings
# (used by retrieve(), below) catches the real google.genai.errors.APIError
# internally and re-raises this instead — it is NOT a subclass of APIError,
# so without also catching it here, an error during retrieval slips past the
# `except APIError` guard below and crashes the app with a raw traceback.
# ChatGoogleGenerativeAI (used by rerank_with_llm) does not do this — it lets
# the original APIError propagate — which is why this gap wasn't caught until
# an error happened to land on the embeddings side of the pipeline.
from langchain_google_genai._common import GoogleGenerativeAIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning_rec.config import (
    CONTENT_FILE,
    EMPLOYEES_FILE,
    INDEX_DIR,
    NUM_RECOMMENDATIONS,
    TOP_K,
    shadowed_dotenv_vars,
)
from learning_rec.index_meta import IndexMetadataError
from learning_rec.llm_utils import api_error_guidance
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


def _has_google_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def _needs_api_key(kind: RetrieverKind, use_rerank: bool) -> bool:
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
    index=2 if _has_google_key() else 1,
    help=(
        "**dense** — FAISS over Gemini embeddings (the thesis baseline, "
        "ported from OpenAI).  \n"
        "**bm25** — sparse lexical scoring, no API calls.  \n"
        "**hybrid** — Reciprocal Rank Fusion of dense + BM25 (recommended)."
    ),
)

use_rerank = st.sidebar.checkbox(
    "LLM re-rank with reasons",
    value=_has_google_key(),
    help=(
        f"Calls {os.getenv('CHAT_MODEL', 'gemini-2.0-flash')} to re-rank and "
        "explain each pick. Free under Gemini's free tier for light use. "
        "Uncheck to see raw retrieval candidates only (no chat-model calls)."
    ),
)

top_k = st.sidebar.slider("Top-K candidates to retrieve", 5, 20, TOP_K)
n_recs = (
    st.sidebar.slider("Recommendations after rerank", 3, 10, NUM_RECOMMENDATIONS)
    if use_rerank
    else top_k
)

# Cost / prereq hint
if _needs_api_key(retriever_kind, use_rerank):
    if not _has_google_key():
        st.sidebar.error(
            "GOOGLE_API_KEY not set. Either set it in your environment / "
            ".env, or pick **bm25** + uncheck **LLM re-rank** for a fully "
            "offline demo."
        )
    # Surfaced before anything else: when a shell export shadows .env, every
    # other diagnostic misleads, because edits to .env have no effect at all.
    _shadowed = shadowed_dotenv_vars()
    if _shadowed:
        st.sidebar.warning(
            f"{', '.join(f'`{n}`' for n in _shadowed)} is set in your shell "
            "and overrides `.env`, so your `.env` value is being ignored. "
            "If authentication fails, this is the first thing to check — run "
            "`echo $GOOGLE_API_KEY` in the terminal you launched from, then "
            "`unset GOOGLE_API_KEY` and restart."
        )

    # Deliberately no key-format validation here. An earlier version warned
    # when the key did not start with `AIza`, on the assumption that was the
    # Gemini key format. It is the *legacy* format: AI Studio now issues
    # service-account-bound "auth keys" (`AQ.` prefix) by default, and
    # standard `AIza` keys are slated for rejection in September 2026. The
    # check therefore flagged correct, current keys as suspect. Prefix
    # sniffing is the wrong tool — the API is the only authority on whether
    # a credential works, and its errors are handled where the calls happen.
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
    elif _needs_api_key(retriever_kind, use_rerank) and not _has_google_key():
        st.error(
            "This configuration needs a Gemini API key. Set `GOOGLE_API_KEY` "
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
                results = rerank_with_llm(emp, candidates, n=n_recs) if use_rerank else None
            except FileNotFoundError as e:
                st.error(
                    f"Could not load the index ({e}). Run "
                    "`python scripts/build_index.py --reset` first."
                )
                st.stop()
            except IndexMetadataError as e:
                # The index was built by a different embedding model. Worth
                # its own branch: the message already says exactly what to
                # do, and this is the one failure here that would otherwise
                # produce plausible-looking but wrong recommendations.
                st.error(str(e))
                st.stop()
            except (APIError, GoogleGenerativeAIError) as e:
                # Guidance is derived from the error itself rather than
                # hardcoded: auth failures, rate limits and genuine outages
                # need opposite advice, and telling someone to "wait and
                # retry" a 401 sends them in circles. See
                # learning_rec.llm_utils.classify_api_error.
                detail = f"{e.code} {e.status}: {e.message}" if isinstance(e, APIError) else str(e)
                st.error(f"Gemini API error — {detail}\n\n{api_error_guidance(e)}")
                st.stop()

            if use_rerank:
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
