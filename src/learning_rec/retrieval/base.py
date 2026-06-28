"""Retriever protocol and the candidate dict shape used everywhere downstream."""

from __future__ import annotations

from typing import Protocol, TypedDict


class Candidate(TypedDict):
    """The dict shape returned by every retriever and consumed by the LLM rerank."""

    content_id: str
    content_name: str
    description: str
    score: float


class Retriever(Protocol):
    """Anything with `.retrieve(query, k)` returning Candidates is a retriever."""

    def retrieve(self, query: str, k: int) -> list[Candidate]: ...
