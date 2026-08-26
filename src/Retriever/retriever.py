from langchain_qdrant import QdrantVectorStore
from src.llm_gateway import embedding_model
import os
from src.state import AgentState
from src.config.settings import settings





async def reterival(state:AgentState):
    qusestion = state["msg"][-1].content


    vector_store = QdrantVectorStore.from_existing_collection(
    embedding= embedding_model(),
    collection_name="cpu_docs",
    url= settings.QDRANT_CLUSTER_URL,
    api_key=settings.QDRANT_API_KEY
    )

    docs =    vector_store.similarity_search(qusestion,k=3)
    context =    "\n\n".join([doc.page_content for doc in docs])
    return {
        "context":context
    }


    




