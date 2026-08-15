"""Retrieval strategies for the recommender.

Three concrete retrievers, all conforming to the `Retriever` protocol:

- `DenseRetriever` — semantic search via FAISS + Gemini embeddings (the
  thesis baseline).
- `BM25Retriever` — sparse lexical search over the same corpus. Catches
  exact technical-term matches (e.g. "Kubernetes", "dbt") that dense
  embeddings smooth over.
- `HybridRetriever` — Reciprocal Rank Fusion (RRF) of any two or more
  retrievers. Robust to score-scale differences without normalization.

A `build_retriever(kind, ...)` factory wires them up from CLI flags.
"""

from learning_rec.retrieval.base import Candidate, Retriever
from learning_rec.retrieval.bm25 import BM25Retriever
from learning_rec.retrieval.dense import DenseRetriever
from learning_rec.retrieval.factory import build_retriever
from learning_rec.retrieval.hybrid import HybridRetriever

__all__ = [
    "BM25Retriever",
    "Candidate",
    "DenseRetriever",
    "HybridRetriever",
    "Retriever",
    "build_retriever",
]
