"""Tests for the offline portions of the recommender (no API calls)."""

from __future__ import annotations

import pandas as pd

from learning_rec.recommender import build_query


class TestBuildQuery:
    def test_skills_are_doubled(self):
        """The skill string must appear twice in the embedding query.

        This is a deliberate retrieval-time weighting: in the thesis
        evaluation, duplicating the skills text in the query nudged FAISS
        toward skill-relevant content without needing a separate reranker
        stage.
        """
        emp = pd.Series(
            {
                "skills": "Python;SQL",
                "job_description": "Data work",
                "strengths": "Communication",
                "interests": "ML",
            }
        )
        q = build_query(emp)
        assert q.count("Python") == 2
        assert q.count("SQL") == 2

    def test_handles_missing_optional_fields(self):
        emp = pd.Series({"skills": "Python", "strengths": "Communication"})
        # job_description and interests missing — should still produce a string
        q = build_query(emp)
        assert isinstance(q, str)
        assert "Python" in q
