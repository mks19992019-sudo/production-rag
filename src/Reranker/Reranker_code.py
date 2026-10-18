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




