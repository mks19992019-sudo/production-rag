from ..Retriever.EVAL_reterival import reterival
from ..state import AgentState


def retrieval_node(state: AgentState):
    query = state['msg'][-1].content
    documents = reterival().invoke(query)
    return {'documents': documents}
