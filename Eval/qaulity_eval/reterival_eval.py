from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from src.Retriever.EVAL_reterival import reterival, Reranker_reterival
from src.llm_gateway import model, GroqJudge, GroqJudge2
import mlflow

GOLDEN_PATH = "goldens/reterival.json"
THRESHOLD = .7
JUDGE1 = GroqJudge()
JUDGE2 = GroqJudge2()





Reterival = reterival()
Reranker = Reranker_reterival()

with open(GOLDEN_PATH) as f:
    golden = json.load(f)


test_cases = []

#print(golden)

for g in golden["test_cases"]:
    

    # reterived = Reterival.invoke(g["query"])
    reterived = Reranker.invoke(g["question"], 5)

    # reterival_context = [doc.page_content for doc in reterived]
    reterival_context = [doc['document'].page_content for doc in reterived]

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            expected_output=g['ideal_answer'],
            retrieval_context=reterival_context,
            actual_output="(generator not evaluated in this run)"
        )
    )


metric = [
    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=JUDGE1,
        include_reason=True
    ),
    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=JUDGE2,
        include_reason=True
    )
]





evaluate(
        test_cases=test_cases,
        metrics=metric,
        async_config=AsyncConfig(
            max_concurrent=1,
            throttle_value=100,
        ),
        display_config=DisplayConfig(
            results_folder="./eval_results"
        ),
    )

  




