from deepeval import evaluate
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric , ContextualRecallMetric
from src.Retriever.EVAL_reterival import reterival,Reranker_reterival
from src.llm_gateway import model , GroqJudge ,GroqJudge2


GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = .7
JUDGE1 = GroqJudge()
JUDGE2 = GroqJudge2()

Reterival = reterival()

with open(GOLDEN_PATH) as f:
    golden = json.load(f)



test_cases = []

#print(golden)

for g in golden:

    #reterived = Reterival.invoke(g["query"])
    reterived = Reranker_reterival().invoke(g["query"],5)
    #
    #reterival_context = [doc.page_content for doc in reterived]
    reterival_context = [doc['document'].page_content for doc in reterived]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g['ideal_answer'],

            retrieval_context= reterival_context,
            actual_output="(generator not evaluated in this run)"

        )
    )

metric = [
    ContextualRecallMetric(threshold=THRESHOLD,model=JUDGE1,include_reason=False),
    ContextualPrecisionMetric(threshold=THRESHOLD,model=JUDGE2,include_reason=False)
]

from deepeval.evaluate import AsyncConfig
evaluate(
    test_cases=test_cases,
    metrics=metric,
    async_config=AsyncConfig(
        max_concurrent=1,
        throttle_value=100
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





