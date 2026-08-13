from langchain_qdrant import QdrantVectorStore
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings



def reterival():
    vector_db =QdrantVectorStore.from_existing_collection(embedding= jinaEmbedding_model(),
        collection_name="cpu_docs",
        url= settings.QDRANT_CLUSTER_URL,
        api_key=settings.QDRANT_API_KEY
        )
    return vector_db
     



print(reterival().similarity_search(k=5,query='hi tell me about cpu'))



