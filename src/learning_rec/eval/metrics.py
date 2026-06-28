"""Pure-function IR metrics. No I/O, no dependencies on the rest of the project.

All three accept an ordered list of retrieved ids (rank 1 first) and the set
of relevant ids, and return a float in [0, 1].

By convention, when the relevant set is empty:
- recall@k returns 1.0 (vacuously all 0 relevant items were retrieved)
- precision@k returns 0.0 (no possible true positives — k > 0 in denominator)
- MRR returns 0.0 (no rank to be reciprocal of)
"""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 1.0
    top_k = list(retrieved)[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant.

    precision@k = |retrieved[:k] ∩ relevant| / k
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def mrr_at_k(retrieved: Iterable[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item in the top-k; 0 if none.

    mrr@k = 1 / rank_of_first_relevant   (with rank starting at 1)
    """
    if not relevant:
        return 0.0
    for rank, cid in enumerate(list(retrieved)[:k], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0
