from deepeval import evaluate
import json
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric , ContextualRecallMetric

from test.test_reterival import reterival



GOLDEN_PATH = "Golden_dataset/retriever_goldens.json"

Reterival = reterival()

with open(GOLDEN_PATH) as f:
    golden = json.load(f)

test_cases = []

for g in golden:
    reterived = Reterival.invoke