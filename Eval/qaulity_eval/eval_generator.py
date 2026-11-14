"""Evaluate generation with faithfulness and answer relevancy."""
import argparse
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from Eval.eval_common import load_examples, save_metric_report
from src.genrator.genrator import generate
from src.Retriever.EVAL_reterival import Reranker_reterival
from src.llm_gateway import GroqJudge, GroqJudge2

def run(top_k: int = 6):
    #retriever = Reranker_reterival()
    cases = []
    examples = load_examples()
    for item in examples:
        #docs = retriever.invoke(item["question"], top_k)
        #context = [doc["document"].page_content for doc in docs]
        cases.append(LLMTestCase(
            input=item["question"],
            actual_output=generate(item["question"]), # ans genrated by llm
            expected_output=item["expected_answer"], # actual answer from the dataset
            context=item["expected_retrieval_output"],
            
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
        ["faithfulness", "answer_relevancy"], "generator", top_k,
    )
    return evaluation_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    run(parser.parse_args().top_k)
