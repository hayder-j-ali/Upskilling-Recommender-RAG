"""Rule-based relevance labels for the evaluation harness.

Without expert-curated labels (this is a portfolio repo, not a production
deployment), we derive ground truth from skill overlap between employee and
content. The rule is intentionally simple and transparent so reviewers can
audit it.

**Definition.** A content item is *relevant* to an employee iff the
case-insensitive intersection of their normalized skill sets has at least
`MIN_SKILL_OVERLAP` elements (default 2).

We require >=2 (not >=1) because single-skill overlap creates obvious false
positives — every "Python" course would be deemed relevant to every employee
who knows Python, which is not how an upskilling recommendation should work.

These labels are a *self-consistent benchmark*, not absolute relevance. They
let us compare retrieval methods (baseline vs hybrid vs reranked) on the same
yardstick. Treat the absolute numbers with the caveat in mind.
"""

from __future__ import annotations

import pandas as pd

from learning_rec.ingest import to_list

MIN_SKILL_OVERLAP = 2


def _normalize(items) -> set[str]:
    return {s.strip().lower() for s in to_list(items) if s.strip()}


def relevant_ids(emp: pd.Series, content: pd.DataFrame) -> set[str]:
    """Return the set of content_ids judged relevant for this employee."""
    emp_skills = _normalize(emp.get("skills", ""))
    if not emp_skills:
        return set()

    relevant: set[str] = set()
    for row in content.itertuples(index=False):
        course_skills = _normalize(row.skills)
        if len(emp_skills & course_skills) >= MIN_SKILL_OVERLAP:
            relevant.add(row.content_id)
    return relevant
