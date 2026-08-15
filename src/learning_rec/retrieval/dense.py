"""Dense semantic retrieval via a pre-built FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from learning_rec import index_meta
from learning_rec.config import EMBEDDING_MODEL, INDEX_DIR
from learning_rec.retrieval.base import Candidate
from learning_rec.vector_store import VectorStore


class DenseRetriever:
    """Wraps a vector store. Constructed once per process and reused."""

    def __init__(self, vectordb: VectorStore, embedder: GoogleGenerativeAIEmbeddings) -> None:
        self._vectordb = vectordb
        self._embedder = embedder

    @classmethod
    def from_index(cls, index_dir: Path = INDEX_DIR) -> DenseRetriever:
        # Before loading: confirm this index was built by the model we are
        # about to query it with. Mismatched models still produce
        # same-dimensioned vectors, so without this the failure is silent.
        index_meta.verify(index_dir, EMBEDDING_MODEL)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        return cls(VectorStore.load(index_dir), embeddings)

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        results = self._vectordb.similarity_search_with_relevance_scores(
            query, self._embedder, k=k
        )
        return [
            Candidate(
                content_id=doc.metadata["cid"],
                content_name=doc.metadata["name"],
                description=doc.page_content[:500],
                score=float(score),
            )
            for doc, score in results
        ]
