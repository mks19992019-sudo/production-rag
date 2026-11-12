"""End-to-end RAG triad: context relevance, faithfulness, answer relevancy."""
import argparse
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from Eval.eval_common import load_examples
from src.llm_gateway import GroqJudge, GroqJudge2
from src.ragpipline.pipline import rag_pipline

def run(top_k: int = 6):
    rag = rag_pipline()
    cases = []
    for item in load_examples():
        result = rag.invoke(item["question"], top_k)
        cases.append(LLMTestCase(
            input=item["question"],
            actual_output=result["ans"],
            expected_output=item["expected_answer"],
            retrieval_context=result["context"],
        ))
    return evaluate(
        test_cases=cases,
        metrics=[
            ContextualRelevancyMetric(threshold=.7, model=GroqJudge(), include_reason=True),
            FaithfulnessMetric(threshold=.7, model=GroqJudge2(), include_reason=True),
            AnswerRelevancyMetric(threshold=.7, model=GroqJudge(), include_reason=True),
        ],
        async_config=AsyncConfig(max_concurrent=1, throttle_value=100),
        display_config=DisplayConfig(results_folder="./eval_results"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    run(parser.parse_args().top_k)
