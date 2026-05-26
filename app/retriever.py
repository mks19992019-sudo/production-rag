from langchain_qdrant import QdrantVectorStore
from llm import embedding_model
import os
from state import AgentState





def reterival(state:AgentState):
    qusestion = state["msg"][-1].content
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model(),
        collection_name="cpu_docs",
        url= os.getenv("QDRANT_URL")
    )

    return   vector_store.similarity_search(qusestion,k=3)




