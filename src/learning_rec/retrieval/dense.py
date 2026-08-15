"""Dense semantic retrieval via a pre-built FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from learning_rec.config import EMBEDDING_MODEL, INDEX_DIR
from learning_rec.retrieval.base import Candidate


class DenseRetriever:
    """Wraps a FAISS vector store. Constructed once per process and reused."""

    def __init__(self, vectordb: FAISS) -> None:
        self._vectordb = vectordb

    @classmethod
    def from_index(cls, index_dir: Path = INDEX_DIR) -> DenseRetriever:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vectordb = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(vectordb)

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        results = self._vectordb.similarity_search_with_relevance_scores(query, k=k)
        return [
            Candidate(
                content_id=doc.metadata["cid"],
                content_name=doc.metadata["name"],
                description=doc.page_content[:500],
                score=float(score),
            )
            for doc, score in results
        ]
