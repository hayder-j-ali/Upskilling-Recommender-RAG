"""Recommendation pipeline composed of two stages:

- `retrieve(emp, retriever, k)` — dense, BM25, or hybrid retrieval (any
  `Retriever` from `learning_rec.retrieval`). Used by the eval harness to
  score retrieval quality independently of the LLM stage.
- `recommend(emp, retriever, ...)` — full pipeline: retrieve, then LLM
  re-rank to a top-N JSON list with justifications.
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_openai import ChatOpenAI

from learning_rec.config import (
    CHAT_MODEL,
    NUM_RECOMMENDATIONS,
    OUTPUT_DIR,
    TEMPERATURE,
    TOP_K,
)
from learning_rec.prompts import BASE_SYSTEM_PROMPT
from learning_rec.retrieval.base import Candidate, Retriever


def build_query(emp: pd.Series) -> str:
    """Build the embedding-search query for an employee.

    Skills are repeated to bias the semantic search toward skill overlap,
    which the offline thesis evaluation found to be the strongest signal.
    The same query is used by BM25 — duplicating skill terms also raises
    their BM25 term frequency, keeping retriever signals aligned.
    """
    skills = "; ".join(str(emp.get("skills", "")).split(";"))
    job = emp.get("job_description", "")
    strengths = "; ".join(str(emp.get("strengths", "")).split(";"))
    interests = emp.get("interests", "")
    return (
        f"Skills: {skills}. {skills}. "
        f"Job: {job}. "
        f"Strengths: {strengths}. "
        f"Interests: {interests}."
    )


def retrieve(emp: pd.Series, retriever: Retriever, k: int = TOP_K) -> list[Candidate]:
    """Run retrieval for an employee using the given strategy."""
    return retriever.retrieve(build_query(emp), k=k)


def rerank_with_llm(
    emp: pd.Series,
    candidates: list[Candidate],
    llm: ChatOpenAI | None = None,
    n: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    """Ask the LLM to re-rank the candidate list to the top-N with reasons."""
    if llm is None:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, max_tokens=512)

    response = llm.invoke(
        [
            {"role": "system", "content": BASE_SYSTEM_PROMPT.format(n=n)},
            {
                "role": "user",
                "content": "EMPLOYEE PROFILE:\n"
                + json.dumps(emp.to_dict(), ensure_ascii=False, default=str),
            },
            {
                "role": "user",
                "content": "CANDIDATES JSON:\n"
                + json.dumps(candidates, ensure_ascii=False),
            },
        ]
    )

    raw = response.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        err_path = OUTPUT_DIR / "last_invalid_response.log"
        err_path.write_text(raw, encoding="utf-8")
        raise ValueError(
            f"LLM returned invalid JSON; raw output saved to {err_path}"
        ) from None


def recommend(
    emp: pd.Series,
    retriever: Retriever,
    llm: ChatOpenAI | None = None,
    top_k: int = TOP_K,
    n: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    """Full pipeline: retrieval, then LLM re-rank to top-N with reasons."""
    candidates = retrieve(emp, retriever, k=top_k)
    return rerank_with_llm(emp, candidates, llm=llm, n=n)
