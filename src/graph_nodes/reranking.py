from ..Reranker.Reranker_code import Rerankers
from ..state import AgentState
from ..config.hyper_para import load_para


def reranking_node(state: AgentState):
    query = state['msg'][-1].content
    reranker = Rerankers()
    reranked_documents = reranker.rerank(
        query=query,
        documents=state['documents'],
        top_k=load_para.rerank_top_k,
    )
    return {'reranked_documents': reranked_documents}
