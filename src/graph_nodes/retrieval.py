from ..Retriever.EVAL_reterival import get_reterival
from ..state import AgentState
from ..config.hyper_para import load_para



def retrieval_node(state: AgentState):
    query = state['msg'][-1].content
    documents = get_reterival(top_k=load_para.top_k).invoke(query)
    return {'documents': documents}
