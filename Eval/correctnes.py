"""Reference-based correctness evaluation for the Career Point University RAG."""
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

correctness_metric = GEval(
    name="cpu_answer_correctness",
    model=GroqJudge(),
    criteria=("Judge whether the answer correctly answers the question about Career Point University "
              "by comparing it with the expected answer. Evaluate factual accuracy, completeness, "
              "and contradictions. Do not reward irrelevant extra information."),
    evaluation_steps=[
        "Identify the exact information requested by the question.",
        "Extract the claims from the generated answer.",
        "Compare the claims with the expected answer, allowing equivalent wording and paraphrases.",
        "Check CPU-specific names, numbers, dates, locations, programmes, and recognition claims.",
        "Check that all important parts of the expected answer are covered.",
        "Penalize contradictions, important invented facts, omissions, and answers to a different question.",
        "Assign a score from 1 to 10 according to the rubric and give a short reason.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Fully correct, complete, relevant, and factually accurate for CPU."),
        Rubric(score_range=(7, 8), expected_outcome="Mostly correct with only a minor omission or imprecision."),
        Rubric(score_range=(4, 6), expected_outcome="Partially correct but missing an important point or containing a CPU detail error."),
        Rubric(score_range=(1, 3), expected_outcome="Substantially incorrect, contradictory, irrelevant, or based on invented CPU facts."),
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
        evaluation = correctness_metric.measure(case)
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
        "metric": "cpu_answer_correctness",
        "top_k": top_k,
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        "total_examples": len(rows),
        "cases": rows,
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"correctness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved correctness evaluation to {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()
    run(args.top_k, args.output_dir)
