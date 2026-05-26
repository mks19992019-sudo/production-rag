from llm import model
from langgraph.prebuilt import create_react_agent
from state import AgentState
from prompt import system_prompt



tools = []

def create_agents(query_msg,reterival):
    agent = create_react_agent(
        model = model,
        tools= [],
        prompt=reterival

    ).invoke({"messages":query_msg})
    return agent


async def agent(state:AgentState):
    query = state['msg']
    reterival = system_prompt.invoke({'context':state['context']})
    result =  create_agents(query,reterival)
    return {'msg':result}
    



    