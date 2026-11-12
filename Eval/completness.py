"""Reference-based completeness evaluation for the Career Point University RAG."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from Eval.eval_common import load_examples
from src.llm_gateway import GroqJudge
from src.ragpipline.pipline import rag_pipline


completeness_metric = GEval(
    name="cpu_answer_completeness",
    model=GroqJudge(),
    criteria=(
        "Judge whether the generated answer covers all important information needed to answer "
        "the question about Career Point University. Compare it with the expected answer. Focus "
        "on coverage of required facts and question parts, not writing style or factual correctness "
        "unless an error causes an important part to be missing."
    ),
    evaluation_steps=[
        "Identify every distinct request or sub-question in the input.",
        "List the important facts and answer components present in the expected output.",
        "Check whether the actual output addresses every request and includes each important fact.",
        "Check CPU-specific details such as names, numbers, dates, locations, programmes, and recognitions when they are required by the question.",
        "Do not penalize equivalent wording, concise paraphrases, or harmless differences in order.",
        "Penalize omitted sub-questions, missing key facts, incomplete lists, and answers that stop before fully addressing the request.",
        "Assign a score from 1 to 10 according to the rubric and give a short reason focused on missing or covered information.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Covers every important part of the question and all key details from the expected answer.",
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Covers the main request and most key details, with only a minor omission.",
        ),
        Rubric(
            score_range=(4, 6),
            expected_outcome="Covers only part of the request or omits one or more important details.",
        ),
        Rubric(
            score_range=(1, 3),
            expected_outcome="Leaves most of the request unanswered or provides almost none of the required information.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)


def run(top_k: int = 6, output_dir: str = "eval_results"):
    rag = rag_pipline()
    rows = []
    for item in load_examples():
        result = rag.invoke(item["question"], top_k)
        case = LLMTestCase(
            input=item["question"],
            actual_output=result["ans"],
            expected_output=item["expected_answer"],
        )
        evaluation = completeness_metric.measure(case)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "score": evaluation.score,
            "reason": evaluation.reason,
        })

    scores = [row["score"] for row in rows if row["score"] is not None]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "golden_quality_eval",
        "metric": "cpu_answer_completeness",
        "top_k": top_k,
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        "total_examples": len(rows),
        "cases": rows,
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved completeness evaluation to {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()
    run(args.top_k, args.output_dir)
