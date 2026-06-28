"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Given any number of underlying retrievers, fetches a (larger) candidate set
from each, then fuses them with:

    RRF(d) = sum over retrievers of 1 / (c + rank_R(d))

where rank is 1-indexed and `c` is a smoothing constant (60 by convention,
from Cormack, Clarke & Buettcher 2009 — "Reciprocal rank fusion outperforms
Condorcet and individual rank learning methods").

RRF is robust to score-scale differences between retrievers (BM25 returns
unbounded log-space scores; dense cosine returns [0, 1]) so we don't need
any normalization tuning.
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_rec.retrieval.base import Candidate, Retriever

RRF_C = 60


class HybridRetriever:
    """Reciprocal Rank Fusion over two or more child retrievers."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fetch_k: int | None = None,
        rrf_c: int = RRF_C,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least 2 child retrievers")
        self._retrievers = list(retrievers)
        self._fetch_k = fetch_k
        self._rrf_c = rrf_c

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        fetch_k = self._fetch_k or max(2 * k, 30)

        # rrf[cid] = accumulated RRF score; lookups[cid] = a Candidate to return
        rrf: dict[str, float] = {}
        lookups: dict[str, Candidate] = {}

        for retriever in self._retrievers:
            results = retriever.retrieve(query, fetch_k)
            for rank, cand in enumerate(results, start=1):
                cid = cand["content_id"]
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self._rrf_c + rank)
                # First time we see this doc, remember its metadata
                lookups.setdefault(cid, cand)

        # Sort by fused score descending; return top-k with fused score
        top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Candidate(
                content_id=cid,
                content_name=lookups[cid]["content_name"],
                description=lookups[cid]["description"],
                score=float(fused),
            )
            for cid, fused in top
        ]
