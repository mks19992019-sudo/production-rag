from typing import TypedDict , Optional ,Annotated

from langchain_core.messages import BaseMessage 
from langgraph.graph import add_messages





# state of our rag agent 
class AgentState ():
    msg : Annotated[list[BaseMessage],add_messages]
    context : str
    documents : list
    reranked_documents : list
    check : Optional[bool]
    answer : str






