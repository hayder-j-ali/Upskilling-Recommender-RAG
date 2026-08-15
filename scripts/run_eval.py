"""CLI: run the evaluation harness against the current FAISS index.

Examples:
    # full eval: retrieval + rerank, no LLM judge
    python scripts/run_eval.py

    # cheap smoke run: 3 employees, retrieval only (skip rerank)
    python scripts/run_eval.py --limit 3 --no-rerank

    # full eval with LLM-as-judge (extra OpenAI calls)
    python scripts/run_eval.py --judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learning_rec.config import (
    CONTENT_FILE,
    EMPLOYEES_FILE,
    INDEX_DIR,
    OUTPUT_DIR,
)
from learning_rec.eval.pipeline import run_eval


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
        help="Retrieval strategy to evaluate (default: dense baseline).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N employees."
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the LLM rerank stage (retrieval-only eval, no chat-model calls).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run LLM-as-judge on the reranked top-N (extra OpenAI calls).",
    )
    args = parser.parse_args()

    report = run_eval(
        employees_path=args.employees,
        content_path=args.content,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        retriever_kind=args.retriever,
        limit=args.limit,
        rerank=not args.no_rerank,
        judge=args.judge,
    )

    print("\nOverall:")
    for stage, metrics in report.overall.items():
        print(f"  {stage}:")
        for metric_name, agg in metrics.items():
            print(f"    {metric_name}: mean={agg['mean']:.3f}  (n={agg['n']})")
    print(f"\nFull report: {args.output_dir / 'eval_report.json'}")
    print(f"Markdown:    {args.output_dir / 'eval_summary.md'}")


if __name__ == "__main__":
    main()
