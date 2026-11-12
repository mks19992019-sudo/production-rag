"""Reference-free scope evaluation for the Career Point University RAG."""
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


scope_metric = GEval(
    name="cpu_answer_scope",
    model=GroqJudge(),
    criteria=(
        "Evaluate whether the generated answer stays within the scope of the user's question "
        "about Career Point University. The answer should address the requested topic directly, "
        "include only useful related information, avoid unrelated CPU facts, and avoid drifting "
        "to other universities, courses, or general topics. This is a reference-free evaluation; "
        "do not compare against an expected answer."
    ),
    evaluation_steps=[
        "Identify the topic, intent, and boundaries of the user's question.",
        "Identify the main claims and topics covered by the generated answer.",
        "Check whether the answer directly addresses the question instead of changing the subject.",
        "Check for unrelated details, unnecessary background, promotional content, or discussion of other institutions.",
        "Check that the answer is neither substantially narrower than the question nor unnecessarily broad.",
        "Do not penalize a short explanation or a directly useful supporting detail.",
        "Assign a score from 1 to 10 according to the rubric and give a short reason focused on scope drift or scope fit.",
    ],
    rubric=[
        Rubric(
            score_range=(9, 10),
            expected_outcome="Directly answers the question and stays completely within scope with no meaningful irrelevant content.",
        ),
        Rubric(
            score_range=(7, 8),
            expected_outcome="Mostly stays within scope, with only a small amount of unnecessary or loosely related information.",
        ),
        Rubric(
            score_range=(4, 6),
            expected_outcome="Partly addresses the question but contains noticeable scope drift or misses the intended focus.",
        ),
        Rubric(
            score_range=(1, 3),
            expected_outcome="Mostly answers a different topic, contains substantial unrelated content, or does not address the question.",
        ),
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
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
        )
        evaluation = scope_metric.measure(case)
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
        "metric": "cpu_answer_scope",
        "reference_free": True,
        "top_k": top_k,
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
    args = parser.parse_args()
    run(args.top_k, args.output_dir)
