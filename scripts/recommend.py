"""CLI: produce recommendations for every employee in the input file.

Usage:
    python scripts/recommend.py
    python scripts/recommend.py --retriever hybrid
    python scripts/recommend.py --employees data/employees.csv --output-dir output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning_rec.config import (  # noqa: E402
    CONTENT_FILE,
    EMPLOYEES_FILE,
    INDEX_DIR,
    OUTPUT_DIR,
)
from learning_rec.recommender import recommend  # noqa: E402
from learning_rec.retrieval import build_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--employees", type=Path, default=EMPLOYEES_FILE)
    parser.add_argument("--content", type=Path, default=CONTENT_FILE)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval strategy (default: dense, the thesis baseline).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N employees (smoke testing).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.employees)
    if args.limit:
        df = df.head(args.limit)

    retriever = build_retriever(
        args.retriever, index_dir=args.index_dir, content_path=args.content
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        recos = recommend(row, retriever)
        emp_id = row["employee_id"]
        out_path = args.output_dir / f"recommend_{emp_id}.json"
        out_path.write_text(
            json.dumps(recos, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {emp_id} -> {out_path}")


if __name__ == "__main__":
    main()
