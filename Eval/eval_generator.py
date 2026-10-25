from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import json
from deepeval.evaluate import AsyncConfig, DisplayConfig
from src.genrator.genrator import generate
from src.llm_gateway import GroqJudge , GroqJudge2


load_dotenv()
JUDGE1 = GroqJudge()
JUDGE2 = GroqJudge2()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"

THRESHOLD = 0.7
with open (GOLDEN_PATH) as g:
    goldens = json.load(g)


def run():
    

    test_cases = []
    for g in goldens:
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE1,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE2,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(test_cases=test_cases, metrics=metrics ,async_config=AsyncConfig(max_concurrent=1,throttle_value=100,))
    return (result)

if __name__ == "__main__":
    run()