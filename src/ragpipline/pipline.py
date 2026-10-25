from src.genrator.genrator import generate
from src.Retriever.EVAL_reterival import Reranker_reterival



class rag_pipline:
    def __init__(self):
        self.rerank_retriver = Reranker_reterival()

    def invoke(self,query,top_k):
        docs = self.rerank_retriver.invoke(query,top_k)
        context = [doc['document'].page_content for doc in docs]

        genrate_ans = generate(query,context)

        return {
            'query': query,
            'context' : context,
            'ans' : genrate_ans
        }



