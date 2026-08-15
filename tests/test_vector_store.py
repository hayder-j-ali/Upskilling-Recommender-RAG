"""Tests for the FAISS-backed vector store.

These use a deterministic fake embedder rather than the API, so the suite
stays offline. The relevance-score assertions pin the exact formula
inherited from langchain-community (`1 - distance / sqrt(2)` over squared
L2 distances) — that is what keeps retrieval rankings, and therefore the
evaluation numbers published in the README, unchanged by the migration.
"""

from __future__ import annotations

import json
import math

import pytest
from langchain_core.documents import Document

from learning_rec.vector_store import DOCS_FILENAME, INDEX_FILENAME, VectorStore


class FakeEmbedder:
    """Maps text to a fixed vector, so distances are exactly predictable."""

    def __init__(self, table: dict[str, list[float]], dim: int = 3) -> None:
        self._table = table
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        return self._table.get(text, [0.0] * self._dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def store_and_embedder():
    docs = [
        Document(page_content="a", metadata={"cid": "A", "name": "Doc A"}),
        Document(page_content="b", metadata={"cid": "B", "name": "Doc B"}),
        Document(page_content="c", metadata={"cid": "C", "name": "Doc C"}),
    ]
    emb = FakeEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
            "near-a": [0.9, 0.1, 0.0],
        }
    )
    return VectorStore.from_documents(docs, emb), emb


class TestBuild:
    def test_len_and_dimension(self, store_and_embedder):
        store, _ = store_and_embedder
        assert len(store) == 3
        assert store.dimension == 3

    def test_rejects_empty_document_list(self):
        with pytest.raises(ValueError, match="zero documents"):
            VectorStore.from_documents([], FakeEmbedder({}))


class TestSearch:
    def test_returns_nearest_document_first(self, store_and_embedder):
        store, emb = store_and_embedder
        results = store.similarity_search_with_relevance_scores("near-a", emb, k=3)
        assert [d.metadata["cid"] for d, _ in results] == ["A", "B", "C"]

    def test_exact_match_scores_one(self, store_and_embedder):
        """Distance 0 -> relevance 1.0, pinning the score formula."""
        store, emb = store_and_embedder
        (_, score), *_ = store.similarity_search_with_relevance_scores("a", emb, k=1)
        assert score == pytest.approx(1.0)

    def test_score_matches_the_inherited_formula(self, store_and_embedder):
        """Orthogonal unit vectors are squared-L2 distance 2 apart, so the
        formula from langchain-community gives 1 - 2/sqrt(2).
        """
        store, emb = store_and_embedder
        results = store.similarity_search_with_relevance_scores("a", emb, k=3)
        _, second_score = results[1]
        assert second_score == pytest.approx(1.0 - 2.0 / math.sqrt(2))

    def test_scores_are_monotonically_non_increasing(self, store_and_embedder):
        store, emb = store_and_embedder
        scores = [s for _, s in store.similarity_search_with_relevance_scores("near-a", emb, k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_corpus_returns_everything_without_padding(
        self, store_and_embedder
    ):
        """FAISS pads short results with index -1; those must be dropped."""
        store, emb = store_and_embedder
        results = store.similarity_search_with_relevance_scores("a", emb, k=99)
        assert len(results) == 3

    def test_k_must_be_positive(self, store_and_embedder):
        store, emb = store_and_embedder
        with pytest.raises(ValueError, match="k must be positive"):
            store.similarity_search_with_relevance_scores("a", emb, k=0)


class TestPersistence:
    def test_round_trip_preserves_results(self, store_and_embedder, tmp_path):
        store, emb = store_and_embedder
        before = store.similarity_search_with_relevance_scores("near-a", emb, k=3)
        store.save(tmp_path)
        after = VectorStore.load(tmp_path).similarity_search_with_relevance_scores(
            "near-a", emb, k=3
        )
        assert [d.metadata["cid"] for d, _ in before] == [
            d.metadata["cid"] for d, _ in after
        ]
        assert [pytest.approx(s) for _, s in before] == [s for _, s in after]

    def test_documents_persist_as_plain_json_not_a_pickle(
        self, store_and_embedder, tmp_path
    ):
        """The wrapper this replaces required allow_dangerous_deserialization
        because it unpickled the index, executing arbitrary code from disk.
        JSON removes that entirely.
        """
        store, _ = store_and_embedder
        store.save(tmp_path)
        payload = json.loads((tmp_path / DOCS_FILENAME).read_text(encoding="utf-8"))
        assert [d["metadata"]["cid"] for d in payload] == ["A", "B", "C"]
        assert not list(tmp_path.glob("*.pkl"))

    def test_load_raises_when_a_file_is_missing(self, store_and_embedder, tmp_path):
        store, _ = store_and_embedder
        store.save(tmp_path)
        (tmp_path / DOCS_FILENAME).unlink()
        with pytest.raises(FileNotFoundError):
            VectorStore.load(tmp_path)

    def test_load_detects_index_document_count_mismatch(
        self, store_and_embedder, tmp_path
    ):
        """A truncated documents.json would otherwise mis-map every hit."""
        store, _ = store_and_embedder
        store.save(tmp_path)
        payload = json.loads((tmp_path / DOCS_FILENAME).read_text(encoding="utf-8"))
        (tmp_path / DOCS_FILENAME).write_text(json.dumps(payload[:2]), encoding="utf-8")
        with pytest.raises(ValueError, match="inconsistent"):
            VectorStore.load(tmp_path)

    def test_save_writes_both_expected_files(self, store_and_embedder, tmp_path):
        store, _ = store_and_embedder
        store.save(tmp_path / "nested")
        assert (tmp_path / "nested" / INDEX_FILENAME).exists()
        assert (tmp_path / "nested" / DOCS_FILENAME).exists()
