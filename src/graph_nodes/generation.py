from langchain.messages import AIMessage

from ..genrator.genrator import generate
from ..state import AgentState


def generation_node(state: AgentState):
    query = state['msg'][-1].content
    history = state['msg'][:-1]
    context = [
        item['document'].page_content
        for item in state['reranked_documents']
    ]
    answer = generate(query=query, context=context, msg=history)
    return {'answer': answer, 'msg': [AIMessage(content=answer)]}
