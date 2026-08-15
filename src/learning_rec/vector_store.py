"""Minimal FAISS-backed vector store, replacing the langchain-community one.

`langchain-community` was sunset on 2026-05-22 (its final release, 0.4.2,
shipped that day), and no official standalone package took over FAISS. The
`langchain-faiss` name on PyPI is *not* LangChain's: it is published by an
unaffiliated account, carries no summary, homepage, or licence, and declares
no dependencies at all — including no dependency on FAISS itself. Adopting it
to escape a deprecation warning would trade a maintenance concern for a
supply-chain one.

This project used four methods of the wrapper, so the wrapper is dropped in
favour of `faiss` directly, which was already a dependency:

    from_documents / save / load / similarity_search_with_relevance_scores

Behaviour is deliberately identical to what it replaces, so retrieval
rankings and the published evaluation numbers stay valid:

- `IndexFlatL2` over un-normalised vectors, matching the previous default
  of `DistanceStrategy.EUCLIDEAN_DISTANCE` with `normalize_L2=False`.
- The same relevance conversion, `1 - distance / sqrt(2)`, taken from
  langchain-community's `_euclidean_relevance_score_fn`. FAISS returns
  squared L2 distances and the original passed them through unchanged, so
  this does too.

One deliberate improvement: documents are persisted as JSON rather than a
pickle. The wrapper's `load_local()` required `allow_dangerous_deserialization=True`
because it unpickles, which executes arbitrary code from the index file.
Nothing here needs that, so the flag and the risk are both gone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from langchain_core.documents import Document

INDEX_FILENAME = "index.faiss"
DOCS_FILENAME = "documents.json"

_SQRT2 = math.sqrt(2.0)


class Embedder(Protocol):
    """The slice of the embeddings interface this store needs."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


class VectorStore:
    """Flat L2 index over document embeddings, with their metadata."""

    def __init__(self, index: faiss.Index, documents: list[Document]) -> None:
        self._index = index
        self._documents = documents

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def dimension(self) -> int:
        return self._index.d

    @classmethod
    def from_documents(
        cls, documents: list[Document], embedder: Embedder
    ) -> VectorStore:
        if not documents:
            raise ValueError("Cannot build a vector store from zero documents")
        vectors = _to_matrix(embedder.embed_documents([d.page_content for d in documents]))
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return cls(index, list(documents))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        payload = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self._documents
        ]
        (directory / DOCS_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        docs_path = directory / DOCS_FILENAME
        for path in (index_path, docs_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
        index = faiss.read_index(str(index_path))
        raw = json.loads(docs_path.read_text(encoding="utf-8"))
        documents = [
            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            for d in raw
        ]
        if index.ntotal != len(documents):
            raise ValueError(
                f"Index holds {index.ntotal} vectors but {len(documents)} documents "
                f"were loaded from {docs_path}; the index directory is inconsistent."
            )
        return cls(index, documents)

    def similarity_search_with_relevance_scores(
        self, query: str, embedder: Embedder, k: int
    ) -> list[tuple[Document, float]]:
        """Top-k documents with relevance in [0, 1], higher being closer."""
        if k <= 0:
            raise ValueError("k must be positive")
        query_vector = _to_matrix([embedder.embed_query(query)])
        # FAISS pads with -1 when k exceeds the corpus size.
        distances, indices = self._index.search(query_vector, min(k, len(self._documents)))
        results: list[tuple[Document, float]] = []
        for distance, position in zip(distances[0], indices[0], strict=True):
            if position < 0:
                continue
            results.append(
                (self._documents[int(position)], 1.0 - float(distance) / _SQRT2)
            )
        return results
