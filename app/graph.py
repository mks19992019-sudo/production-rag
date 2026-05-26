from langgraph.graph import StateGraph , END , START
from .retriever import reterival
from .agent import agent
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import os
from .state import AgentState
from contextlib import AbstractAsyncContextManager
import asyncio
from langgraph.checkpoint.base import BaseCheckpointSaver
from dotenv import load_dotenv

load_dotenv()



# global_variable 


_CHECKPOINTER_CONTEXT : AbstractAsyncContextManager[AsyncRedisSaver] | None = None
_CHECKPOINTER : AsyncRedisSaver | None = None
_Workflow = None
_resorce_lock = asyncio.Lock()



async def initialize_resources() -> None:
    global _CHECKPOINTER , _CHECKPOINTER_CONTEXT , _Workflow

    if _CHECKPOINTER is not None and _CHECKPOINTER_CONTEXT is not None:
        return
    
    async with _resorce_lock:
        if _CHECKPOINTER is None:
            _CHECKPOINTER_CONTEXT = AsyncRedisSaver.from_conn_string(os.getenv('REDIS_URL'))
            _CHECKPOINTER = await _CHECKPOINTER_CONTEXT.__aenter__()

            await _CHECKPOINTER.setup()

        if _Workflow is None:
            _Workflow = build_workflow(_CHECKPOINTER)

            


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node('retriever',reterival)
    graph.add_node('agent',agent)

    graph.add_edge(START,'retriever')
    graph.add_edge('retriever','agent')

    return graph



def build_workflow(checkpointer: BaseCheckpointSaver):

    return  build_graph().compile(checkpointer)

    

async def  get_workflow():
    await initialize_resources()

    return _Workflow


















