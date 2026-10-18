import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification

MODEL_NAME = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)


model.eval()


query = "What is machine learning?"


documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Machine learning algorithms learn patterns from data.",
    "Python is a programming language.",
    "The weather is very hot today.",
    "Deep learning uses neural networks to learn representations."
]


pairs = [
    [query, document]
    for document in documents
]


print(pairs)