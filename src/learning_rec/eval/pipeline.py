"""Orchestrate a full evaluation run and produce a report.

Stages:
1. Build ground-truth relevance labels for every employee.
2. For each employee, run retrieval (and optionally LLM rerank).
3. Compute per-employee metrics: recall@k, MRR, precision@k.
4. Optionally run LLM-as-judge over the top-5 reranked items.
5. Aggregate to overall means and write `output/eval_report.json` plus a
   short Markdown summary suitable for pasting into the README.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from learning_rec.config import (
    CONTENT_FILE,
    EMPLOYEES_FILE,
    INDEX_DIR,
    NUM_RECOMMENDATIONS,
    OUTPUT_DIR,
    TOP_K,
)
from learning_rec.eval.ground_truth import relevant_ids
from learning_rec.eval.judge import judge_recommendation
from learning_rec.eval.metrics import mrr_at_k, precision_at_k, recall_at_k
from learning_rec.ingest import load_content
from learning_rec.recommender import rerank_with_llm, retrieve
from learning_rec.retrieval import Retriever, build_retriever
from learning_rec.retrieval.factory import RetrieverKind


@dataclass
class EmployeeResult:
    employee_id: str
    relevant_count: int
    retrieval: dict = field(default_factory=dict)        # recall@k, mrr@k, precision@k
    rerank: dict | None = None                            # precision@n for the reranked top-N
    judge: dict | None = None                             # mean score + per-item list


@dataclass
class EvalReport:
    config: dict
    overall: dict
    per_employee: list[dict]


def _aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "n": 0}
    return {"mean": statistics.fmean(values), "n": len(values)}


def evaluate(
    *,
    employees: pd.DataFrame,
    content: pd.DataFrame,
    retriever: Retriever,
    top_k: int = TOP_K,
    n_recs: int = NUM_RECOMMENDATIONS,
    rerank: bool = True,
    judge: bool = False,
) -> EvalReport:
    per_employee: list[EmployeeResult] = []

    for _, emp in employees.iterrows():
        truth = relevant_ids(emp, content)
        candidates = retrieve(emp, retriever, k=top_k)
        retrieved_ids_top_k = [c["content_id"] for c in candidates]

        result = EmployeeResult(
            employee_id=emp["employee_id"],
            relevant_count=len(truth),
            retrieval={
                f"recall@{top_k}": recall_at_k(retrieved_ids_top_k, truth, top_k),
                f"mrr@{top_k}": mrr_at_k(retrieved_ids_top_k, truth, top_k),
                f"precision@{top_k}": precision_at_k(retrieved_ids_top_k, truth, top_k),
            },
        )

        if rerank:
            reranked = rerank_with_llm(emp, candidates, n=n_recs)
            reranked_ids = [r["content_id"] for r in reranked]
            result.rerank = {
                f"precision@{n_recs}": precision_at_k(reranked_ids, truth, n_recs),
            }

            if judge:
                judged = [judge_recommendation(emp, r) for r in reranked]
                result.judge = {
                    "mean_score": statistics.fmean(j["score"] for j in judged) if judged else 0.0,
                    "per_item": judged,
                }

        per_employee.append(result)

    # Aggregate
    def collect(path: list[str]) -> list[float]:
        values = []
        for r in per_employee:
            cur: object = r
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key, None)
                if cur is None:
                    break
            if isinstance(cur, (int, float)):
                values.append(float(cur))
        return values

    overall = {
        "retrieval": {
            f"recall@{top_k}": _aggregate(collect(["retrieval", f"recall@{top_k}"])),
            f"mrr@{top_k}": _aggregate(collect(["retrieval", f"mrr@{top_k}"])),
            f"precision@{top_k}": _aggregate(collect(["retrieval", f"precision@{top_k}"])),
        },
    }
    if rerank:
        overall["rerank"] = {
            f"precision@{n_recs}": _aggregate(collect(["rerank", f"precision@{n_recs}"])),
        }
    if judge:
        overall["judge"] = {"mean_score": _aggregate(collect(["judge", "mean_score"]))}

    config = {
        "top_k": top_k,
        "n_recs": n_recs,
        "rerank": rerank,
        "judge": judge,
        "n_employees": len(per_employee),
        "n_content": len(content),
        "retriever": getattr(retriever, "__class__", type(retriever)).__name__,
    }
    return EvalReport(
        config=config,
        overall=overall,
        per_employee=[asdict(r) for r in per_employee],
    )


def run_eval(
    *,
    employees_path: Path = EMPLOYEES_FILE,
    content_path: Path = CONTENT_FILE,
    index_dir: Path = INDEX_DIR,
    output_dir: Path = OUTPUT_DIR,
    retriever_kind: RetrieverKind = "dense",
    limit: int | None = None,
    rerank: bool = True,
    judge: bool = False,
) -> EvalReport:
    """Load everything, run evaluation, write report files. Returns the report."""
    employees = pd.read_csv(employees_path)
    if limit:
        employees = employees.head(limit)
    content = load_content(content_path)
    retriever = build_retriever(
        retriever_kind, index_dir=index_dir, content_path=content_path
    )

    report = evaluate(
        employees=employees,
        content=content,
        retriever=retriever,
        rerank=rerank,
        judge=judge,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_report.json"
    md_path = output_dir / "eval_summary.md"
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def _markdown_summary(report: EvalReport) -> str:
    cfg = report.config
    lines = [
        "# Evaluation Summary",
        "",
        f"- Employees: {cfg['n_employees']}",
        f"- Content items: {cfg['n_content']}",
        f"- top_k (retrieval): {cfg['top_k']}",
        f"- n_recs (rerank): {cfg['n_recs']}",
        "",
        f"## Retrieval ({cfg['retriever']})",
        "",
        "| Metric | Mean |",
        "| --- | --- |",
    ]
    for name, agg in report.overall["retrieval"].items():
        lines.append(f"| {name} | {agg['mean']:.3f} |")

    if "rerank" in report.overall:
        lines += ["", "## LLM rerank", "", "| Metric | Mean |", "| --- | --- |"]
        for name, agg in report.overall["rerank"].items():
            lines.append(f"| {name} | {agg['mean']:.3f} |")

    if "judge" in report.overall:
        lines += [
            "",
            "## LLM-as-judge (mean relevance score, 0–1)",
            "",
            f"- mean: {report.overall['judge']['mean_score']['mean']:.3f}",
        ]

    return "\n".join(lines) + "\n"
