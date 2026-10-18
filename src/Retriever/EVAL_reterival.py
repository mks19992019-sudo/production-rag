from langchain_qdrant import QdrantVectorStore
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings
from qdrant_client import QdrantClient
import logfire


client = QdrantClient(
    url = settings.QDRANT_CLUSTER_URL,
    api_key=settings.QDRANT_API_KEY
    
)



def reterival():
    vector_db =QdrantVectorStore.from_existing_collection(
        embedding= jinaEmbedding_model(),
        collection_name="LLM eval",
        url= settings.QDRANT_CLUSTER_URL,
        api_key=settings.QDRANT_API_KEY
        )
    logfire.info('fetch is done')
    return vector_db.as_retriever(search_kwargs={'k':5})



#client.delete_collection('cpu_docs')
#print('delete vector database')

     



#print(reterival().similarity_search(k=2,query='what is eval'))




