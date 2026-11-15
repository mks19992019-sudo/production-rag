from langchain_qdrant import QdrantVectorStore
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings
from qdrant_client import QdrantClient
import logfire
from src.Reranker.Reranker_code import Reranker_model
from langsmith import traceable

RERANEKR = None
RETRIVER = None
VECTOR_DB = None




client = QdrantClient(
    url = settings.QDRANT_CLUSTER_URL,
    api_key=settings.QDRANT_API_KEY    
)


@traceable(name="reterival_loading", display_name="load_reterival" )
def get_reterival(top_k=15):
    global VECTOR_DB , RETRIVER
    if RETRIVER is None:
        vector_db =QdrantVectorStore.from_existing_collection(
            embedding= jinaEmbedding_model(),
            collection_name="CPU RAG",
            url= settings.QDRANT_CLUSTER_URL,
            api_key=settings.QDRANT_API_KEY
            )
        logfire.info('fetchting the data.... from db ')
    
        RETRIVER = vector_db.as_retriever(search_kwargs={'k': top_k})
    return RETRIVER













class Reranker_reterival:
    def __init__(self,top_k,R_top_k):
        global RERANEKR
        if RERANEKR is None:
            RERANEKR = Reranker_model()

        self.R_top_k = R_top_k

        self.Reranker = RERANEKR
        self.reterival = get_reterival(top_k)
    def invoke(self,query):


        documents = self.reterival.invoke(query)
        logfire.info(
            "Documents retrieved",
            count=len(documents),
        )

        results = self.Reranker.rerank(
            query=query,
            documents=documents,
            top_n=self.R_top_k,
        )

        return results



    







     








