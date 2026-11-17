from src.genrator.genrator import generate 
from src.config.hyper_para import  load_para
from src.Retriever.EVAL_reterival import Reranker_reterival




class pipline_eval:
    def __init__(self):
        self.rerank_retriver = Reranker_reterival(load_para.top_k,load_para.rerank_top_k)

    def invoke(self,query,msg_history=None):
        docs = self.rerank_retriver.invoke(query)
        context = [doc.page_content for doc in docs]

        genrate_ans = generate(query=query,context=context,msg=msg_history)

        return {
            'query': query,
            'context' : context,
            'ans' : genrate_ans        
        }



