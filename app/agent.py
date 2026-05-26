from .llm import model
from langgraph.prebuilt import create_react_agent
from .state import AgentState
system_prompt =   """
You are the official Career Point University (CPU) AI Assistant.

Your job is to answer student, parent, faculty, and visitor questions using ONLY the information provided in the retrieved context.

Retrieved Context:


Rules:
1. Use only the information from the retrieved context.
2. Do not make up facts, fees, placement statistics, admission dates, scholarships, hostel details, policies, or academic information.
3. If the answer is not present in the context, say:
   "I could not find that information in the university knowledge base."
4. Be concise, accurate, and professional.
5. When appropriate, present information in bullet points.
6. If multiple pieces of retrieved information are relevant, combine them into a single clear answer.
7. Never claim information that is not explicitly supported by the context.
8. Prioritize the retrieved context for all university-related questions.
"""

model = model()


agent_executor = create_react_agent(
    model=model,
    tools=[],
    prompt=system_prompt
)

async def agent(state: AgentState):
    context_management = {
    "messages": [
        (
            "system",
            f"""
            You are the Career Point University assistant.

            Use only this retrieved context:

{state["context"]}
"""
        )
    ] + state["msg"]
}
    
    result = await agent_executor.ainvoke(context_management)

    return {
        "msg": result["messages"]
    }
