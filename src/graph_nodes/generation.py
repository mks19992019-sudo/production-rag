from langchain.messages import AIMessage

from ..genrator.genrator import generate
from ..state import AgentState


def generation_node(state: AgentState):
    query = state['msg'][-1].content
    history = state['msg'][:-1]
    context = [
        document.page_content
        for document in state['reranked_documents']
    ]
    answer = generate(query=query, context=context, msg=history)
    return {'msg': [AIMessage(content=answer)]}
