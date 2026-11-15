from ..Reranker.Reranker_code import Reranker_model
from ..state import AgentState
from ..config.hyper_para import load_para

RERANKER = None




def reranking_node(state: AgentState):
    global RERANKER
    if RERANKER is None:
        RERANKER = Reranker_model()
    query = state['msg'][-1].content

    reranker = RERANKER
    reranked_documents = reranker.rerank(
        query=query,
        documents=state['documents'],
        top_k=load_para.rerank_top_k,
    )
    return {'reranked_documents': reranked_documents}
