"""Tests for the BM25 retriever (offline — no API)."""

from __future__ import annotations

from langchain_core.documents import Document

from learning_rec.retrieval.bm25 import BM25Retriever, tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Python SQL") == ["python", "sql"]

    def test_keeps_code_relevant_punctuation(self):
        # C++, C#, .NET should survive as single tokens
        assert "c++" in tokenize("learn C++ for systems work")
        assert "c#" in tokenize("C# basics")
        assert ".net" in tokenize("intro to .NET")

    def test_strips_other_punctuation(self):
        # Commas and semicolons are not tokens themselves
        assert tokenize("Python; SQL, R") == ["python", "sql", "r"]


class TestBM25Retriever:
    def _make(self) -> BM25Retriever:
        docs = [
            Document(
                page_content="kubernetes deployment scaling rollouts",
                metadata={"cid": "K1", "name": "K8s Ops"},
            ),
            Document(
                page_content="introduction to python programming language",
                metadata={"cid": "P1", "name": "Python Basics"},
            ),
            Document(
                page_content="advanced python data structures and patterns",
                metadata={"cid": "P2", "name": "Advanced Python"},
            ),
            Document(
                page_content="figma component libraries design systems",
                metadata={"cid": "F1", "name": "Figma DS"},
            ),
        ]
        return BM25Retriever.from_documents(docs)

    def test_rare_term_finds_exact_doc(self):
        """BM25's killer use case: rare lexical match (e.g. tool names)."""
        r = self._make()
        results = r.retrieve("kubernetes scaling", k=1)
        assert results[0]["content_id"] == "K1"

    def test_python_query_ranks_python_docs_above_others(self):
        r = self._make()
        results = r.retrieve("python data structures", k=4)
        top_ids = [c["content_id"] for c in results[:2]]
        assert "P2" in top_ids
        assert "F1" not in top_ids  # design doc should NOT be in top-2

    def test_returns_candidate_shape(self):
        r = self._make()
        results = r.retrieve("python", k=2)
        for c in results:
            assert set(c.keys()) == {"content_id", "content_name", "description", "score"}
            assert isinstance(c["score"], float)
