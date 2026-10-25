from deepeval.metrics import faithfulness, contextual_relevancy , answer_relevancy
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from src.ragpipline.pipline import rag_pipline
import json
from src.llm_gateway import GroqJudge , GroqJudge2


# we are using dataset for just taking the query not for refernce eval and we dont need golden dataset here

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
THRESHOLD = 0.7
JUDGE_MODEL = GroqJudge()
JUDGE_MODEL2 = GroqJudge2()



with open(GOLDEN_PATH) as f:
    goldens = json.load(f)



def pipline_eval(rag):
    test_cases = []


    for g in goldens:

        result = rag.invoke(g['query'],6)

        test_cases.append(
            LLMTestCase(
                input=g['query'],
                retrieval_context=result['context'],
                actual_output=result['ans'],

                
            )
        )
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL2, include_reason=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=False),
    ]
    evaluate(test_cases=test_cases, metrics=metrics)



if __name__ == "__main__": 
    pipline_eval(rag_pipline())