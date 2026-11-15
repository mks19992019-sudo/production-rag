#import torch
#from transformers import AutoTokenizer , AutoModelForSequenceClassification
from pyclbr import Class

from langsmith import traceable
import logfire
import cohere
from src.config.settings import settings




class Reranker_model:
    def __init__(self,model_name="rerank-v5.0-pro"):

        self.clinet = cohere.Client(settings.COHERE_API_KEY) 


        self.model_name = model_name
    def rerank(self, query, documents, top_n=6):
        if not documents:
            return []
        if documents and isinstance(documents[0], str):
            texts = documents
        else:
            texts = [doc.page_content for doc in documents]

        response = self.clinet.rerank(query=query,documents=texts,top_n=min(top_n,len(documents)))

        rank_docs = []
        count = 0

        for result in response.results:
            


            ## cohere retuen the index of original data so we can easly merge it
            original_docs = documents[result.index]




            rank_docs.append(original_docs)
        return rank_docs























#MODEL_NAME = "BAAI/bge-reranker-base"
#MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""
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


        pairs = [[query,doc.page_content if hasattr(doc, "page_content") else doc] for doc in documents]

        scores = []

        for batch_index, start in enumerate(range(0, len(pairs), 8)):

            batch = pairs[start:start + 8]

            # Tokenization
            inputs = self.tokenizer(batch,padding=True,truncation=True,max_length=256,return_tensors="pt").to(self.device)

            # Model inference
                
            with torch.inference_mode():
                logits = self.model(**inputs).logits.squeeze(-1)

            scores.extend(logits.cpu().tolist())

            # Ranking

            ranked = sorted(zip(documents, scores),key=lambda item: item[1],reverse=True)
            results = [{"document": doc,"score": score}for doc, score in ranked[:top_k]]

            return results

"""

