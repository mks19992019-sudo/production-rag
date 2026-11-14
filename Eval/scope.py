"""Reference-free scope evaluation for the Career Point University RAG."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from Eval.eval_common import DEFAULT_SCOPE_DATASET, load_examples, normalize_scope_example
from src.llm_gateway import GroqJudge
from src.pipline.pipline import pipline_eval

scope_metric = GEval(
    name="cpu_answer_scope",
    model=GroqJudge(),
    criteria=(
        "Evaluate whether the generated answer performs the expected action and satisfies the "
        "success criteria for the user's input. Use the expected outcome as the baseline. "
        "Judge semantic equivalence rather than exact wording, and penalize missing, incorrect, "
        "or unrelated content."
    ),
    evaluation_steps=[
        "Identify the expected action and each requirement in the success criteria.",
        "Compare the generated answer with the expected action and success criteria.",
        "Do not penalize equivalent wording, concise answers, or directly useful supporting details.",
        "Penalize omitted requirements, wrong actions, contradictions, and unrelated content.",
        "Assign a score from 1 to 10 and give a short reason focused on satisfied or missed requirements.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Performs the expected action and satisfies all success criteria.",
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Performs the expected action and satisfies almost all success criteria, with a minor omission.",
        ),
        Rubric(
            score_range=(4, 6),
            expected_outcome="Partly performs the expected action or misses one or more important success criteria.",
        ),
        Rubric(
            score_range=(1, 3),
            expected_outcome="Does not perform the expected action or fails most success criteria.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)


def run(top_k: int = 6,
    output_dir: str = "eval_results",
    dataset_name: str | None = None,
    throttle_seconds: int = 5,
):
    rag = pipline_eval()
    dataset_name, examples = load_examples(
        dataset_name or os.getenv("SCOPE_LANGSMITH_DATASET_NAME", DEFAULT_SCOPE_DATASET),
        normalizer=normalize_scope_example,
        required_fields=("question", "expected_action", "success_criteria"),
        return_dataset_name=True,
    )
    test_cases = []
    for item in examples:
        result = rag.invoke(item["question"], top_k)
        test_cases.append(LLMTestCase(
            input=item["question"],
            actual_output=result["ans"],
            expected_output=f"Expected Action: {item['expected_action']} Success Criteria: {item['success_criteria']}",
        ))

    evaluation_result = evaluate(
        test_cases,
        [scope_metric],
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=1,
            throttle_value=throttle_seconds,
        ),
    )

    rows = []
    for item, test_result in zip(examples, evaluation_result.test_results):
        evaluation = test_result.metrics_data[0]
        rows.append({
            "id": item["id"],
            "input": item["question"],
            "expected_action": item["expected_action"],
            "success_criteria": item["success_criteria"],
            "score": evaluation.score,
            "reason": evaluation.reason,
        })

    scores = [row["score"] for row in rows if row["score"] is not None]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "metric": "cpu_answer_scope",
        "reference_free": False,
        "top_k": top_k,
        "throttle_seconds": throttle_seconds,
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        "total_examples": len(rows),
        "cases": rows,
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"scope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved scope evaluation to {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--dataset-name", default=None, help="LangSmith scope golden dataset name")
    parser.add_argument("--throttle-seconds", type=int, default=5)
    args = parser.parse_args()
    run(args.top_k, args.output_dir, args.dataset_name, args.throttle_seconds)
