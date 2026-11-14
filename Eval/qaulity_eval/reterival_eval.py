"""Evaluate retriever quality using the LangSmith golden dataset."""
import argparse
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from Eval.eval_common import load_examples, save_metric_report
from src.Retriever.EVAL_reterival import Reranker_reterival
from src.llm_gateway import GroqJudge, GroqJudge2

def run(top_k, R_top_k):
    retriever = Reranker_reterival(top_k,R_top_k)
    cases = []
    examples = load_examples()
    for item in examples:
        docs = retriever.invoke(item["question"])
        cases.append(LLMTestCase(
            input=item["question"],
            actual_output="retrieval-only evaluation",
            expected_output=item["expected_answer"],
            retrieval_context=[doc["document"].page_content for doc in docs],

        ))
    evaluation_result = evaluate(
        test_cases=cases,
        metrics=[
            ContextualRecallMetric(threshold=.7, model=GroqJudge(), include_reason=True),
            ContextualPrecisionMetric(threshold=.7, model=GroqJudge2(), include_reason=True),
        ],
        async_config=AsyncConfig(max_concurrent=1, throttle_value=100),
        display_config=DisplayConfig(results_folder="./eval_results"),
    )
    save_metric_report(
        evaluation_result, examples,
        ["contextual_recall", "contextual_precision"],
        "retrieval", top_k,
    )
    return evaluation_result


if __name__ == "__main__":
 
    run(8,4)
