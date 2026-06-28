"""In-memory BM25 retrieval — sparse lexical search.

BM25 complements dense embeddings by giving full weight to exact term matches.
Dense embeddings tend to smooth over rare technical tokens ("Kubernetes",
"dbt", "OpenTelemetry") that are exactly the strongest signals an upskilling
recommender has to work with — fusing BM25 in catches these cases.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from learning_rec.config import CONTENT_FILE
from learning_rec.ingest import build_documents, load_content
from learning_rec.retrieval.base import Candidate

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumerics + a few code-relevant punctuators (C++, C#, .NET)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    """Holds the corpus in memory and serves BM25-ranked candidates per query."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._tokenized = [tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def from_documents(cls, documents: list[Document]) -> BM25Retriever:
        return cls(documents)

    @classmethod
    def from_content_file(cls, content_path: Path = CONTENT_FILE) -> BM25Retriever:
        df: pd.DataFrame = load_content(content_path)
        return cls(build_documents(df))

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # argsort descending; take top-k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Candidate(
                content_id=self._documents[i].metadata["cid"],
                content_name=self._documents[i].metadata["name"],
                description=self._documents[i].page_content[:500],
                score=float(scores[i]),
            )
            for i in ranked_indices
        ]
