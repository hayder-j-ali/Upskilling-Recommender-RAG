"""Pure-function tests for IR metrics. Deterministic, no I/O."""

from __future__ import annotations

import pytest

from learning_rec.eval.metrics import mrr_at_k, precision_at_k, recall_at_k


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_overlap(self):
        assert recall_at_k(["a", "x", "b", "y"], {"a", "b", "c", "d"}, k=4) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_k_smaller_than_relevant_set(self):
        assert recall_at_k(["a", "x"], {"a", "b", "c", "d"}, k=2) == pytest.approx(0.25)

    def test_truncates_to_k(self):
        # the relevant "a" is at rank 5; with k=3 it shouldn't count
        assert recall_at_k(["x", "y", "z", "w", "a"], {"a"}, k=3) == 0.0

    def test_empty_relevant_is_vacuously_one(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 1.0


class TestPrecisionAtK:
    def test_all_top_k_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c", "d"}, k=3) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert precision_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=0)


class TestMrrAtK:
    def test_first_position(self):
        assert mrr_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0

    def test_second_position(self):
        assert mrr_at_k(["x", "a", "b"], {"a"}, k=3) == pytest.approx(0.5)

    def test_third_position(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / 3)

    def test_only_first_match_counts(self):
        # "a" at rank 2, "b" at rank 4 — only the first reciprocal matters
        assert mrr_at_k(["x", "a", "y", "b"], {"a", "b"}, k=4) == pytest.approx(0.5)

    def test_no_match_in_top_k(self):
        assert mrr_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_match_beyond_k_does_not_count(self):
        assert mrr_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert mrr_at_k(["a", "b"], set(), k=2) == 0.0
