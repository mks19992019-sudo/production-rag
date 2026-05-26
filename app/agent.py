from .llm import model
from langgraph.prebuilt import create_react_agent
from .state import AgentState
from .tool import webserch
from .prompt import system_prompt

model = model()


agent_executor = create_react_agent(
    model=model,
    tools=[webserch],
    prompt=system_prompt
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
        "msg": result["messages"]
    }
