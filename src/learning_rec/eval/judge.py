"""LLM-as-judge for end-to-end recommendation quality.

For each (employee, recommended item) pair, ask an LLM to rate relevance on a
4-point ordinal scale. The mean numeric score across all pairs is a coarse
signal of whether the LLM rerank stage is producing genuinely useful results,
independent of the rule-based ground truth in `ground_truth.py`.

This is intentionally separated from IR metrics — it answers a different
question ("is this a good recommendation?") and burns more OpenAI tokens, so
it's opt-in via `scripts/run_eval.py --judge`.
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_openai import ChatOpenAI

from learning_rec.config import CHAT_MODEL, TEMPERATURE

SCALE = {
    "highly_relevant": 1.0,
    "relevant": 0.66,
    "somewhat_relevant": 0.33,
    "not_relevant": 0.0,
}

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a learning-content recommender.

You will be given an employee profile and a single recommended learning item.
Rate the recommendation on this 4-point scale:

  - "highly_relevant"   — directly addresses a primary skill, job duty, or stated interest
  - "relevant"          — clearly useful, addresses a secondary skill or interest
  - "somewhat_relevant" — tangentially related but not a strong fit
  - "not_relevant"      — no meaningful connection

Return ONLY a JSON object:
  {"rating": "<one of the four labels>", "justification": "<one short sentence>"}
"""


def judge_recommendation(
    emp: pd.Series, recommendation: dict, llm: ChatOpenAI | None = None
) -> dict:
    """Judge one recommendation. Returns {"rating", "justification", "score"}."""
    if llm is None:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, max_tokens=200)

    response = llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "EMPLOYEE PROFILE:\n"
                    + json.dumps(emp.to_dict(), ensure_ascii=False, default=str)
                    + "\n\nRECOMMENDED ITEM:\n"
                    + json.dumps(recommendation, ensure_ascii=False)
                ),
            },
        ]
    )

    raw = response.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "rating": "not_relevant",
            "justification": "judge returned invalid JSON",
            "score": 0.0,
        }

    rating = parsed.get("rating", "not_relevant")
    return {
        "rating": rating,
        "justification": parsed.get("justification", ""),
        "score": SCALE.get(rating, 0.0),
    }
