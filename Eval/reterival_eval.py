from deepeval import evaluate
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric , ContextualRecallMetric

from test.test_reterival import reterival
from src.llm_gateway import model , GroqJudge




GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = .7
JUDGE = GroqJudge()

Reterival = reterival()

with open(GOLDEN_PATH) as f:
    golden = json.load(f)

test_cases = []

#print(golden)

for g in golden:

    reterived = Reterival.invoke(g["query"])

    reterival_context = [doc.page_content for doc in reterived]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g['ideal_answer'],
            retrieval_context= reterival_context,
            actual_output="(generator not evaluated in this run)"

        )
    )

metric = [
    ContextualRecallMetric(threshold=THRESHOLD,model=JUDGE,include_reason=False),
    ContextualPrecisionMetric(threshold=THRESHOLD,model=JUDGE,include_reason=False)
]

from deepeval.evaluate import AsyncConfig
evaluate(
    test_cases=test_cases,
    metrics=metric,
    async_config=AsyncConfig(
        max_concurrent=2,
        throttle_value=1
    ),

    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" when you swap it in
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 5,
        "judge_model": 'JUDGE',
        "golden_set": GOLDEN_PATH,
    }
)





