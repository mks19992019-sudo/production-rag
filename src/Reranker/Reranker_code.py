import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification

#MODEL_NAME = "BAAI/bge-reranker-base"
#MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Rerankers:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

        self.device = torch.device('cpu')

        self.model.to(self.device)

    def rerank(self,query,documents,top_k=4):
        if not documents:
            return []
        pairs = [[query, doc.page_content if hasattr(doc, "page_content") else doc] for doc in documents
]

       

        scores = []

        for start in range(0, len(pairs),8):
            batch = pairs[start:start+8]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation = True,
                max_length = 256,
                return_tensors = 'pt'
            )
            with torch.inference_mode():
                logits = self.model(**inputs).logits.squeeze(-1)

            scores.extend(logits.cpu().tolist())

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {"document": doc, "score": score}
            for doc, score in ranked[:top_k]
        ]


## testing
reranker = Rerankers()

query = "What is machine learning?"

documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Python is a programming language used for software development.",
    "Machine learning algorithms learn patterns from data.",
    "The human heart pumps blood throughout the body.",
    "Deep learning is a subset of machine learning."
]

results = reranker.rerank(
    query=query,
    documents=documents,
    top_k=3
)

for rank, item in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {item['score']:.4f}")
    print(f"Document: {item['document']}")

