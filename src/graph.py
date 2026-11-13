from langgraph.graph import StateGraph , END , START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from .config.settings import settings
from .state import AgentState
from contextlib import AbstractAsyncContextManager
import asyncio
from langgraph.checkpoint.base import BaseCheckpointSaver
from .graph_nodes.Guardrails import Guardrails_agent
from .graph_nodes.retrieval import retrieval_node
from .graph_nodes.reranking import reranking_node
from .graph_nodes.generation import generation_node



# global_variable 
_CHECKPOINTER_CONTEXT : AbstractAsyncContextManager[AsyncPostgresSaver] | None = None
_CHECKPOINTER : AsyncPostgresSaver | None = None
_Workflow = None
_resorce_lock = asyncio.Lock()



async def initialize_resources() -> None:
    global _CHECKPOINTER , _CHECKPOINTER_CONTEXT , _Workflow

    if _CHECKPOINTER is not None and _CHECKPOINTER_CONTEXT is not None:
        return
    
    async with _resorce_lock:
        if _CHECKPOINTER is None:
            _CHECKPOINTER_CONTEXT = AsyncPostgresSaver.from_conn_string(
                settings.DATABASE_URL_POOLED
            )
            _CHECKPOINTER = await _CHECKPOINTER_CONTEXT.__aenter__()

            await _CHECKPOINTER.setup()

        if _Workflow is None:
            _Workflow = build_workflow(_CHECKPOINTER)

            

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('Guardrails',Guardrails_agent)
    graph.add_node('retrieval', retrieval_node)
    graph.add_node('reranking', reranking_node)
    graph.add_node('generation', generation_node)
    
    graph.add_edge(START,'Guardrails')
    graph.add_conditional_edges(
        'Guardrails',
        route_guardrail,
        {'retrieval':'retrieval', END:END},
    )
    graph.add_edge('retrieval','reranking')
    graph.add_edge('reranking','generation')
    graph.add_edge('generation',END)

    return graph


async def route_guardrail(state:AgentState):
    if state['check']:
        return "retrieval"
    return END


def build_workflow(checkpointer: BaseCheckpointSaver):

    return  build_graph().compile(checkpointer)

    

async def  get_workflow():
    await initialize_resources()

    return _Workflow

async def get_checkpointer():
    await initialize_resources()
    return _CHECKPOINTER


async def close_graph_resources() -> None:
    global _CHECKPOINTER_CONTEXT, _CHECKPOINTER, _Workflow

    async with _resorce_lock:
        if _CHECKPOINTER_CONTEXT is not None:
            await _CHECKPOINTER_CONTEXT.__aexit__(None, None, None)

        _CHECKPOINTER_CONTEXT = None
        _CHECKPOINTER = None
        _Workflow = None















