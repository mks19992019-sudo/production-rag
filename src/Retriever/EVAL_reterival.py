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

def reterival(top_k=15):
    vector_db =QdrantVectorStore.from_existing_collection(
        embedding= jinaEmbedding_model(),
        collection_name="CPU RAG",
        url= settings.QDRANT_CLUSTER_URL,
        api_key=settings.QDRANT_API_KEY
        )
    logfire.info('fetchting the data.... from db ')
    
    return vector_db.as_retriever(search_kwargs={'k': top_k})



class Reranker_reterival:
    def __init__(self,top_k,R_top_k):

        self.R_top_k = R_top_k
        self.Reranker = Rerankers()
        self.reterival = reterival(top_k)
    def invoke(self,query):
        documents = self.reterival.invoke(query)
        logfire.info(
            "Documents retrieved",
            count=len(documents),
        )

        results = self.Reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.R_top_k,
        )

        return results



    







     








