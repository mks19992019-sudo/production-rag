from typing import TypedDict , Optional
from langchain_core.messages import BaseMessage





# state of our rag agent 
class AgentState (TypedDict):
    msg : BaseMessage
    context : str
    ans : str





