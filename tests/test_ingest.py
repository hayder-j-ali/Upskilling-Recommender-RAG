"""Tests for the catalogue ingestion layer.

These tests deliberately exercise only the offline path — no API calls.
The embedding/index build is covered by manual smoke runs documented in the
README, not CI (would require an OpenAI key and burn quota on every push).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from learning_rec.ingest import build_documents, load_content, to_list

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class TestToList:
    """`to_list` must handle every shape a CSV/Excel cell can plausibly be."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Python;SQL", ["Python", "SQL"]),
            ("Python, SQL", ["Python", "SQL"]),
            ('["Python", "SQL"]', ["Python", "SQL"]),
            (["Python", "SQL"], ["Python", "SQL"]),
            ("", []),
            ("[]", []),
            ("[None]", []),
            ([], []),
            ("  Python  ;  SQL  ", ["Python", "SQL"]),
            ("Python", ["Python"]),
        ],
    )
    def test_normalization(self, raw, expected):
        assert to_list(raw) == expected

    def test_nan_returns_empty(self):
        assert to_list(math.nan) == []

    def test_list_with_none_is_dropped(self):
        assert to_list(["Python", None, "SQL"]) == ["Python", "SQL"]


class TestLoadContent:
    """Round-trip the committed synthetic catalogue."""

    def test_loads_and_normalizes(self):
        df = load_content(DATA_DIR / "learning_content.csv")
        assert len(df) > 0
        # skills/keywords must come out as Python lists after to_list
        assert isinstance(df.iloc[0]["skills"], list)
        assert isinstance(df.iloc[0]["keywords"], list)
        # required columns present
        for col in [
            "content_id",
            "content_name",
            "content_description",
            "duration_seconds",
        ]:
            assert col in df.columns


class TestBuildDocuments:
    def test_one_doc_per_row(self):
        df = pd.DataFrame(
            [
                {
                    "content_id": "X1",
                    "content_name": "Foo",
                    "content_description": "A course about foo.",
                    "content_language": "en",
                    "duration_seconds": 3600,
                    "skills": ["Python", "SQL"],
                    "keywords": ["foo", "bar"],
                }
            ]
        )
        docs = build_documents(df)
        assert len(docs) == 1
        d = docs[0]
        assert d.metadata["cid"] == "X1"
        assert d.metadata["duration"] == 3600
        # tags merge keywords + skills with no duplicates
        assert set(d.metadata["tags"]) == {"Python", "SQL", "foo", "bar"}
        # page content includes description + tags
        assert "foo" in d.page_content and "Python" in d.page_content

    def test_invalid_duration_becomes_none(self):
        df = pd.DataFrame(
            [
                {
                    "content_id": "X1",
                    "content_name": "Foo",
                    "content_description": "desc",
                    "content_language": "en",
                    "duration_seconds": "not a number",
                    "skills": [],
                    "keywords": [],
                }
            ]
        )
        docs = build_documents(df)
        assert docs[0].metadata["duration"] is None
