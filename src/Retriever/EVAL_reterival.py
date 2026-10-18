from langchain_qdrant import QdrantVectorStore
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings
from qdrant_client import QdrantClient
import logfire
from src.Reranker.Reranker_code import Rerankers


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
    return vector_db.as_retriever(search_kwargs={'k':15})



class Reranker_reterival:
    def __init__(self):
        self.Reranker = Rerankers()
        self.reterival = reterival()
    def invoke(self,query,top_k):
        documents = self.reterival.invoke(query)
        logfire.info(
            "Documents retrieved",
            count=len(documents),
        )

        results = self.Reranker.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        return results



    









#client.delete_collection('cpu_docs')
#print('delete vector database')

     








