"""Evaluate generation with faithfulness and answer relevancy."""
import argparse
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from Eval.eval_common import load_examples, save_metric_report
from src.genrator.genrator import generate
from src.llm_gateway import GroqJudge, GroqJudge2

def run():
    cases = []
    examples = load_examples(dataset_name="cpu_homepage_golden_dataset")
    for item in examples:
        print(item["expected_retrieval_output"])
       
        cases.append(LLMTestCase(
            input=item["question"],
            
            actual_output=generate(query=item["question"],context=item['expected_retrieval_output']), # ans genrated by llm
            retrieval_context=item["expected_retrieval_output"] if isinstance(item["expected_retrieval_output"], list) else [item["expected_retrieval_output"]] # deep eval dont know from where our retrival comes so that why we use it bec faithfullnes meteric have need this retrival context
            
        ))
    evaluation_result = evaluate(
        test_cases=cases,
        metrics=[
            FaithfulnessMetric(threshold=.7, model=GroqJudge(), include_reason=True),
            AnswerRelevancyMetric(threshold=.7, model=GroqJudge2(), include_reason=True),
        ],
        async_config=AsyncConfig(max_concurrent=1, throttle_value=100),
        display_config=DisplayConfig(results_folder="./eval_results"),
    )
    save_metric_report(
        evaluation_result, examples,
        ["faithfulness", "answer_relevancy"], "generator",
    )
    return evaluation_result

if __name__ == "__main__":

    run()
