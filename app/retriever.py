from langchain_qdrant import QdrantVectorStore
from .llm import embedding_model
import os
from .state import AgentState





async def reterival(state:AgentState):
    qusestion = state["msg"][-1].content
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model(),
        collection_name="cpu_docs",
        url= os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    except:
        vector_store = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model(),
        collection_name="cpu_docs",
        url= os.getenv("QDRANT_URL_LOCAL"),
        
    )


    docs =    vector_store.similarity_search(qusestion,k=3)
    context =    "\n\n".join([doc.page_content for doc in docs])
    return {
        "context":context
    }

    




