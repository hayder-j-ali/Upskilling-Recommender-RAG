"""Evaluation harness for the recommender.

- `ground_truth`: rule-based relevance labels derived from skill overlap.
- `metrics`: classic IR metrics (recall@k, MRR, precision@k) as pure functions.
- `judge`: LLM-as-judge for end-to-end recommendation quality.
- `pipeline`: orchestrates a full evaluation run and writes a report.
"""
