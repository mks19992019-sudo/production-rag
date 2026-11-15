import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from langsmith import traceable
import logfire

#MODEL_NAME = "BAAI/bge-reranker-base"
#MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"



class Reranker_model:

    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = torch.device("cpu")

        # Track model initialization separately
        with logfire.span("Load Reranker Model",model_name=model_name,device=str(self.device)):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self.model.eval()
            self.model.to(self.device)

        logfire.info("Reranker initialized",model_name=model_name,device=str(self.device))

    @traceable(name="Rerank",run_type="chain",tags=["rag", "reranking"])
    def rerank(self, query, documents, top_k=6):

        if not documents:
            return []

        with logfire.span("Reranking",document_count=len(documents),top_k=top_k):
            pairs = [[query,doc.page_content if hasattr(doc, "page_content") else doc] for doc in documents]

            scores = []

            for batch_index, start in enumerate(range(0, len(pairs), 8)):

                batch = pairs[start:start + 8]

                # Tokenization
                with logfire.span("Reranker Tokenization",batch_index=batch_index,batch_size=len(batch)):
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt"
                    ).to(self.device)

                # Model inference
                with logfire.span("Reranker Inference",batch_index=batch_index,batch_size=len(batch)):
                    with torch.inference_mode():
                        logits = self.model(**inputs).logits.squeeze(-1)

                    scores.extend(logits.cpu().tolist())

            # Ranking
            with logfire.span("Sort Ranked Documents"):
                ranked = sorted(zip(documents, scores),key=lambda item: item[1],reverse=True)

                results = [{"document": doc,"score": score}for doc, score in ranked[:top_k]]

            logfire.info("Reranking completed",input_documents=len(documents),output_documents=len(results))

            return results



