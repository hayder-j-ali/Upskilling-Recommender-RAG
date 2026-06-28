"""Tests for the hybrid (RRF) retriever. Uses fake child retrievers — no API calls."""

from __future__ import annotations

import pytest

from learning_rec.retrieval.base import Candidate
from learning_rec.retrieval.hybrid import RRF_C, HybridRetriever


class FakeRetriever:
    """Returns a fixed ranked list, ignoring the query."""

    def __init__(self, ranked_ids: list[str]) -> None:
        self._ranked_ids = ranked_ids

    def retrieve(self, query: str, k: int) -> list[Candidate]:  # noqa: ARG002
        return [
            Candidate(
                content_id=cid,
                content_name=f"Course {cid}",
                description=f"Desc of {cid}",
                score=1.0 / (i + 1),
            )
            for i, cid in enumerate(self._ranked_ids[:k])
        ]


class TestHybridRetriever:
    def test_requires_at_least_two_retrievers(self):
        with pytest.raises(ValueError):
            HybridRetriever([FakeRetriever(["a"])])

    def test_rrf_math_matches_formula(self):
        """RRF(d) = sum over R of 1 / (c + rank_R(d))."""
        r1 = FakeRetriever(["a", "b", "c"])
        r2 = FakeRetriever(["b", "a", "c"])
        # Use a small c so the math is easy to inspect, and fetch_k large enough
        # to see all 3 docs from both retrievers.
        hybrid = HybridRetriever([r1, r2], fetch_k=3, rrf_c=10)
        result = hybrid.retrieve("anything", k=3)

        # a: rank 1 in r1, rank 2 in r2 -> 1/(10+1) + 1/(10+2) = 1/11 + 1/12
        # b: rank 2 in r1, rank 1 in r2 -> 1/(10+2) + 1/(10+1) = same as a
        # c: rank 3 in both              -> 1/(10+3) + 1/(10+3) = 2/13
        scores = {c["content_id"]: c["score"] for c in result}
        assert scores["a"] == pytest.approx(1 / 11 + 1 / 12)
        assert scores["b"] == pytest.approx(1 / 11 + 1 / 12)
        assert scores["c"] == pytest.approx(2 / 13)

    def test_rewards_consistent_top_ranks(self):
        """A doc ranked highly by both retrievers should beat one ranked highly by only one."""
        r1 = FakeRetriever(["x", "y", "z", "a"])
        r2 = FakeRetriever(["x", "y", "z", "a"])
        hybrid = HybridRetriever([r1, r2], fetch_k=4)
        top = hybrid.retrieve("anything", k=2)
        assert [c["content_id"] for c in top] == ["x", "y"]

    def test_fuses_disjoint_lists(self):
        """When retrievers disagree, union appears in fused output."""
        r1 = FakeRetriever(["a", "b"])
        r2 = FakeRetriever(["c", "d"])
        hybrid = HybridRetriever([r1, r2], fetch_k=2)
        result = hybrid.retrieve("anything", k=4)
        ids = {c["content_id"] for c in result}
        assert ids == {"a", "b", "c", "d"}

    def test_returns_at_most_k(self):
        r1 = FakeRetriever([f"d{i}" for i in range(20)])
        r2 = FakeRetriever([f"d{i}" for i in range(20)])
        hybrid = HybridRetriever([r1, r2], fetch_k=20)
        assert len(hybrid.retrieve("q", k=5)) == 5

    def test_default_rrf_c_is_60(self):
        assert RRF_C == 60
