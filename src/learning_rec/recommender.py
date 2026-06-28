"""For a given employee profile, retrieve candidate courses and ask an LLM to re-rank."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from learning_rec.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    INDEX_DIR,
    NUM_RECOMMENDATIONS,
    OUTPUT_DIR,
    TEMPERATURE,
    TOP_K,
)
from learning_rec.prompts import BASE_SYSTEM_PROMPT


def load_vector_store(index_dir: Path = INDEX_DIR) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=200)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_query(emp: pd.Series) -> str:
    """Build the embedding-search query for an employee.

    Skills are repeated to bias the semantic search toward skill overlap,
    which the offline thesis evaluation found to be the strongest signal.
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


def recommend(
    emp: pd.Series,
    vectordb: FAISS,
    llm: ChatOpenAI | None = None,
    top_k: int = TOP_K,
    n: int = NUM_RECOMMENDATIONS,
) -> list[dict]:
    if llm is None:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, max_tokens=512)

    query = build_query(emp)
    retrieved = vectordb.similarity_search_with_relevance_scores(query, k=top_k)
    candidates = [
        {
            "content_id": doc.metadata["cid"],
            "content_name": doc.metadata["name"],
            "description": doc.page_content[:500],
            "score": float(score),
        }
        for doc, score in retrieved
    ]

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
