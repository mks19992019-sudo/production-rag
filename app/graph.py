from langgraph.graph import StateGraph , END , START
from state import State
import asyncio
import retriever
import agent



async def build_graph(State):
    graph = StateGraph(State)

    graph.add_node('retriever',retriever)
    graph.add_node('agent',agent)

    graph.add_edge(START,'retriever')
    graph.add_edge('retriever','agent')

    return graph











