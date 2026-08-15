"""Build a FAISS vector store from the learning-content catalogue.

Source schema (CSV or XLSX) — see data/learning_content.csv:
    content_id, content_name, content_description, content_language,
    duration_seconds, keywords, skills

`keywords` and `skills` may be stored as semicolon-separated strings,
comma-separated strings, or Python-literal lists — `to_list()` normalizes them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import tqdm
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from learning_rec import index_meta
from learning_rec.config import CONTENT_FILE, EMBEDDING_MODEL, INDEX_DIR
from learning_rec.vector_store import VectorStore


def to_list(x) -> list[str]:
    """Normalize a cell that may be a list, semicolon/comma string, or empty."""
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip() and str(v).lower() != "none"]
    if pd.isna(x) or x in ("", "[]", "[None]"):
        return []
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return [
                    str(v).strip()
                    for v in val
                    if str(v).strip() and str(v).lower() != "none"
                ]
        except (ValueError, SyntaxError):
            pass
        return [t.strip() for t in re.split(r"[;,]", x) if t.strip()]
    return [str(x).strip()]


def load_content(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.fillna("")
    df["skills"] = df["skills"].apply(to_list)
    df["keywords"] = df["keywords"].apply(to_list)
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for r in tqdm.tqdm(df.itertuples(index=False), total=len(df), desc="Building docs"):
        text = " ".join(
            [
                str(r.content_description),
                " ".join(r.keywords or []),
                " ".join(r.skills or []),
            ]
        )
        duration = getattr(r, "duration_seconds", None)
        try:
            duration = int(duration) if str(duration).strip().isdigit() else None
        except (TypeError, ValueError):
            duration = None
        meta = {
            "cid": r.content_id,
            "name": r.content_name,
            "language": getattr(r, "content_language", ""),
            "duration": duration,
            "tags": list(set((r.keywords or []) + (r.skills or []))),
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def build_vector_store(
    content_path: Path = CONTENT_FILE,
    index_dir: Path = INDEX_DIR,
    reset: bool = False,
) -> VectorStore:
    df = load_content(content_path)
    docs = build_documents(df)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if reset and index_dir.exists():
        for p in index_dir.glob("*"):
            p.unlink()

    vectordb = VectorStore.from_documents(docs, embeddings)
    vectordb.save(index_dir)
    # Recorded so DenseRetriever can refuse an index built by a different
    # model — see learning_rec.index_meta for why that check has to exist.
    index_meta.write(
        index_dir, embedding_model=EMBEDDING_MODEL, n_documents=len(docs)
    )
    return vectordb
