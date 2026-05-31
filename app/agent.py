from .llm import model
#from langgraph.prebuilt import create_react_agent.   '''(in this the middleware not suport)'''
from .state import AgentState
from .tool import webserch
from .prompt import system_prompt
from langchain.agents.middleware import PIIMiddleware
from langchain.agents import create_agent

model = model()


agent_executor = create_agent(
    model=model,
    tools=[webserch],
    system_prompt=system_prompt,
    middleware=[
        PIIMiddleware(
            "email",
            strategy='redact',
            apply_to_input=True
        ),
        PIIMiddleware(
            'credit_card',
            apply_to_input=True,
            strategy='mask'

        ),
        PIIMiddleware(
            'ip',
            strategy='hash',
            apply_to_input=True
        ),PIIMiddleware(
            'mac_address',
            apply_to_input=True,
            strategy='mask'
        )
    ],

    
)


async def agent(state: AgentState):
    context_management = {

    "messages": [
        (
            "system",
            f"""
Retrieved Context:
{state["context"]}
Use this context when answering the user's question.
If the context is insufficient, you may use available tools.
"""        )
    ] + state["msg"]
}
    
    result = await agent_executor.ainvoke(context_management)

    return {
        "msg": result["messages"],
        'answer' : result["messages"][-1].content
    }
