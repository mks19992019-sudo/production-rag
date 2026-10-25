
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from src.Retriever.EVAL_reterival import reterival, Reranker_reterival
from src.llm_gateway import model, GroqJudge, GroqJudge2
import mlflow

GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = .7
JUDGE1 = GroqJudge()
JUDGE2 = GroqJudge2()


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Reterival_componet_eval")


Reterival = reterival()
Reranker = Reranker_reterival()

with open(GOLDEN_PATH) as f:
    golden = json.load(f)


test_cases = []

#print(golden)

for g in golden:

    # reterived = Reterival.invoke(g["query"])
    reterived = Reranker.invoke(g["query"], 6)

    # reterival_context = [doc.page_content for doc in reterived]
    reterival_context = [doc['document'].page_content for doc in reterived]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
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


with mlflow.start_run(run_name="minilm-reranker") as run:

    # 1. Record configuration
    mlflow.log_params({
        "retriever": "qdrant_with_minilm_reranker",
        "embedding_model": "Jina",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "retrieval_top_k": 12,
        "reranker_top_k": 6,
        "chunk_size": 800,
        "chunk_overlap": 100,
        "threshold": THRESHOLD,
        "golden_set": GOLDEN_PATH,
        "num_test_cases": len(test_cases),
        "judge_recall": type(JUDGE1).__name__,
        "judge_precision": type(JUDGE2).__name__,
    })

    # 2. Run evaluation and save DeepEval report
    result = evaluate(
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

    # 3. Log average metric scores to MLflow
    # DeepEval calculates the individual scores.
    # This small section aggregates them for MLflow.

    metric_scores = {}

    for test_result in result.test_results:
        for metric_result in test_result.metrics_data:
            if metric_result.score is not None:
                name = metric_result.name.lower().replace(" ", "_")
                metric_scores.setdefault(name, []).append(
                    float(metric_result.score)
                )

    for name, scores in metric_scores.items():
        if scores:
            mlflow.log_metric(name, sum(scores) / len(scores))

    # 4. Save DeepEval reports as MLflow artifacts
    mlflow.log_artifacts("./eval_results", artifact_path="deepeval_results")

    print("MLflow run ID:", run.info.run_id)
    print("Evaluation completed.")



