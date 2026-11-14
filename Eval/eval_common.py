"""Shared dataset loading and normalization for evaluation scripts."""
from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langsmith import Client
import logfire

logfire.configure()

load_dotenv()

DEFAULT_DATASET = "cpu_homepage_golden_dataset"
DEFAULT_SCOPE_DATASET = "CPU_RAG_Scope_Evaluation_15_Questions"

def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return "" if value is None else str(value)

def normalize_example(inputs: dict[str, Any], outputs: dict[str, Any] | None = None) -> dict[str, Any]:
    outputs = outputs or {}
    question = _first(inputs, "input", "question", "query")
    answer = _first(outputs, "expected_output", "ideal_answer", "expected_answer", "answer", "output")
    retrieval_reference = _first(
        outputs,
        "expected_reterival_output",  # schema used by golden_quality_eval
        "expected_retrieval_output",
        "expected_context",
        "reference_context",
        "context",
        "contexts",
    )

    success_criteria = _first(outputs, "success_criteria")

    expected_action = _first(outputs, "expected_action")

    if isinstance(retrieval_reference, str):


        retrieval_reference = [retrieval_reference]

    return {
        "id": _first(inputs, "id") or _first(outputs, "id"),
        "question": question,
        "expected_answer": _as_text(answer),
        "expected_retrieval_output": _as_text(retrieval_reference),
        "expected_action": _as_text(expected_action),
        "success_criteria": _as_text(success_criteria)
    }


def load_examples(dataset_name) -> list[dict[str, Any]]:
    """Load only from the configured LangSmith dataset; never use local goldens."""


   
    examples = list(Client().list_examples(dataset_name=dataset_name))
    if not examples:
        logfire.info('golden dataset not found in online langsmith ')
        
        raise RuntimeError(f"LangSmith dataset '{dataset_name}' contains no examples")
    normalized = [normalize_example(e.inputs or {}, e.outputs or {}) for e in examples]
    missing = [i for i, e in enumerate(normalized, 1) if not e["question"]]

    if missing:
        logfire.info('LangSmith examples missing question/query input at rows')

        raise ValueError(f"LangSmith examples missing question/query input at rows: {missing}")
    
    return normalized















def save_metric_report(
    evaluation_result: Any,
    examples: list[dict[str, Any]],
    metric_names: list[str],
    report_name: str,
    top_k: int,
    output_dir: str = "eval_results",
) -> dict[str, Any]:
    """Persist DeepEval scores in the same shape as the scope evaluation."""
    rows = []
    for index, (item, test_result) in enumerate(zip(examples, evaluation_result.test_results)):
        metrics = {}
        for metric_name, metric_data in zip(metric_names, test_result.metrics_data):
            metrics[metric_name] = {
                "score": metric_data.score,
                "reason": metric_data.reason,
                "success": metric_data.success,
            }
        rows.append({
            "id": item.get("id") or index,
            "input": item["question"],
            "metrics": metrics,
        })

    averages = {}
    for metric_name in metric_names:
        scores = [
            row["metrics"][metric_name]["score"]
            for row in rows
            if row["metrics"].get(metric_name, {}).get("score") is not None
        ]
        averages[metric_name] = round(sum(scores) / len(scores), 4) if scores else None

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "golden_quality_eval",
        "metric": metric_names,
        "top_k": top_k,
        "average_score": averages,
        "total_examples": len(rows),
        "cases": rows,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output = output_path / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {report_name} evaluation to {output}")
    return report
