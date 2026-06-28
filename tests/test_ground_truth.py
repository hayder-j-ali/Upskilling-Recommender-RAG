"""Tests for the rule-based ground-truth labeller."""

from __future__ import annotations

import pandas as pd

from learning_rec.eval.ground_truth import relevant_ids

CONTENT = pd.DataFrame(
    [
        # 3 skill overlaps with Alex below — definitely relevant
        {"content_id": "C1", "skills": "Python;SQL;Airflow"},
        # 2 overlaps — relevant (at the threshold)
        {"content_id": "C2", "skills": "Python;SQL;Leadership"},
        # 1 overlap — NOT relevant under the 2-overlap rule
        {"content_id": "C3", "skills": "Python;Java;C++"},
        # 0 overlaps — not relevant
        {"content_id": "C4", "skills": "Figma;UX;Design"},
        # case difference — should still match
        {"content_id": "C5", "skills": "python;sql;ETL"},
    ]
)


class TestRelevantIds:
    def test_picks_courses_with_enough_overlap(self):
        emp = pd.Series({"skills": "Python;SQL;Airflow;ETL"})
        result = relevant_ids(emp, CONTENT)
        assert result == {"C1", "C2", "C5"}

    def test_case_insensitive(self):
        emp = pd.Series({"skills": "PYTHON;sql;airflow"})
        result = relevant_ids(emp, CONTENT)
        assert "C1" in result  # 3 overlaps ignoring case
        assert "C5" in result  # also case-difference match

    def test_employee_with_no_skills_returns_empty(self):
        emp = pd.Series({"skills": ""})
        assert relevant_ids(emp, CONTENT) == set()

    def test_employee_with_one_overlapping_skill_returns_empty(self):
        # Only "Python" overlaps with anything — but only by 1, below threshold
        emp = pd.Series({"skills": "Python"})
        assert relevant_ids(emp, CONTENT) == set()
