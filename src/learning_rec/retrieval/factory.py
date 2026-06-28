"""Single entry point for constructing a retriever from a string flag."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from learning_rec.config import CONTENT_FILE, INDEX_DIR
from learning_rec.retrieval.base import Retriever
from learning_rec.retrieval.bm25 import BM25Retriever
from learning_rec.retrieval.dense import DenseRetriever
from learning_rec.retrieval.hybrid import HybridRetriever

RetrieverKind = Literal["dense", "bm25", "hybrid"]


def build_retriever(
    kind: RetrieverKind,
    *,
    index_dir: Path = INDEX_DIR,
    content_path: Path = CONTENT_FILE,
) -> Retriever:
    if kind == "dense":
        return DenseRetriever.from_index(index_dir)
    if kind == "bm25":
        return BM25Retriever.from_content_file(content_path)
    if kind == "hybrid":
        return HybridRetriever(
            [
                DenseRetriever.from_index(index_dir),
                BM25Retriever.from_content_file(content_path),
            ]
        )
    raise ValueError(f"Unknown retriever kind: {kind!r}")
