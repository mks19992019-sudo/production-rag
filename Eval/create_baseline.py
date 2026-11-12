"""Create the initial JSON baseline from the LangSmith golden dataset.

This is a snapshot artifact. LangSmith remains the source of detailed traces,
while this file gives regression checks a stable local reference point.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from Eval.eval_common import load_examples
from src.Retriever.EVAL_reterival import Reranker_reterival
from src.genrator.genrator import generate
from src.llm_gateway import GroqJudge, GroqJudge2


TOP_K = int(os.getenv("EVAL_TOP_K", "6"))
OUTPUT_PATH = Path(os.getenv("BASELINE_PATH", "eval_results/baseline.json"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def score(metric, case: LLMTestCase) -> dict:
    result = metric.measure(case)
    return {"score": result.score, "reason": result.reason}


def average(rows: list[dict], metric_name: str) -> float | None:
    values = [row[metric_name]["score"] for row in rows if row[metric_name]["score"] is not None]
    return round(sum(values) / len(values), 4) if values else None


def run() -> dict:
    examples = load_examples()
    retriever = Reranker_reterival()
    retrieval_metrics = {
        "contextual_precision": ContextualPrecisionMetric(threshold=.7, model=GroqJudge2()),
        "contextual_recall": ContextualRecallMetric(threshold=.7, model=GroqJudge()),
    }
    generation_metrics = {
        "faithfulness": FaithfulnessMetric(threshold=.7, model=GroqJudge()),
        "answer_relevancy": AnswerRelevancyMetric(threshold=.7, model=GroqJudge2()),
    }
    pipeline_metrics = {
        "contextual_relevancy": ContextualRelevancyMetric(threshold=.7, model=GroqJudge()),
        "faithfulness": FaithfulnessMetric(threshold=.7, model=GroqJudge2()),
        "answer_relevancy": AnswerRelevancyMetric(threshold=.7, model=GroqJudge()),
    }

    retrieval_rows, generation_rows, pipeline_rows = [], [], []
    for item in examples:
        docs = retriever.invoke(item["question"], TOP_K)
        context = [doc["document"].page_content for doc in docs]
        answer = generate(item["question"], context)

        retrieval_case = LLMTestCase(
            input=item["question"],
            actual_output="retrieval-only evaluation",
            expected_output=item["expected_retrieval_output"] or item["expected_answer"],
            retrieval_context=context,
        )
        generation_case = LLMTestCase(
            input=item["question"], actual_output=answer,
            expected_output=item["expected_answer"], retrieval_context=context,
        )
        retrieval_rows.append({"id": item["id"], **{name: score(metric, retrieval_case) for name, metric in retrieval_metrics.items()}})
        generation_rows.append({"id": item["id"], **{name: score(metric, generation_case) for name, metric in generation_metrics.items()}})
        pipeline_rows.append({"id": item["id"], **{name: score(metric, generation_case) for name, metric in pipeline_metrics.items()}})

    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": os.getenv("LANGSMITH_DATASET_NAME", "golden_quality_eval"),
        "git_commit": git_commit(),
        "top_k": TOP_K,
        "levels": {
            "retrieval": {"averages": {name: average(retrieval_rows, name) for name in retrieval_metrics}, "cases": retrieval_rows},
            "generator": {"averages": {name: average(generation_rows, name) for name in generation_metrics}, "cases": generation_rows},
            "pipeline": {"averages": {name: average(pipeline_rows, name) for name in pipeline_metrics}, "cases": pipeline_rows},
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Baseline written to {OUTPUT_PATH}")
    return baseline


if __name__ == "__main__":
    run()
