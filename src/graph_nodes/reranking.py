from ..Reranker.Reranker_code import Rerankers
from ..state import AgentState


def reranking_node(state: AgentState):
    query = state['msg'][-1].content
    reranker = Rerankers()
    reranked_documents = reranker.rerank(
        query=query,
        documents=state['documents'],
        top_k=6,
    )
    return {'reranked_documents': reranked_documents}
