from src.llm_gateway import model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from prompt import prompt

# Build the generator chain
chain = prompt | model() | StrOutputParser()



def generate(query: str, context: list[str],msg=None) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)


    return chain.invoke({"question": query, "context": context_text ,'chat_history':msg})




