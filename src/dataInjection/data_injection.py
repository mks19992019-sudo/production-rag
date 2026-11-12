from langchain_community.document_loaders import WebBaseLoader , TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings

import logfire

from qdrant_client import QdrantClient
import asyncio

logfire.configure()



client = QdrantClient(
        url=settings.QDRANT_CLUSTER_URL,
        api_key=settings.QDRANT_API_KEY
    )


def extract():
    loader = TextLoader("Data/raw/cpur_data.txt")
    return loader.load()



def chunking_semantic(documents):

    chunker = SemanticChunker(
        embeddings=jinaEmbedding_model(),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=10,
        min_chunk_size=50,
    )


    return chunker.split_documents(documents)




def chunking_Recursive_text_split(documents):

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
        chunk_overlap = 50,
        separators=["\n\n","\n",". ","  ",""]

    )

    chunker = text_spliter.split_documents(documents)

    return chunker


# here .from_documents automatically set dimesion of vector store based on embedding model dimension
# in our case it 2048 
# interally 
'''client.create_collection(
    collection_name="cpu_docs",
    vectors_config=VectorParams(
        size=2048,
        distance=Distance.COSINE,
    ),
)'''



def vector_db():
    logfire.info('loading the data and prepare the docs')
    docs = extract()
    logfire.info('start chucking of data')
    chunks = chunking_semantic(docs)


    
    logfire.info('creating cluster in qudrant with name of CPU RAG')
    vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=jinaEmbedding_model(),
            url = settings.QDRANT_CLUSTER_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name= "CPU RAG",
            

        )

    return vector_store



async def initialize_vectorstore():
    #client.delete_collection("CPU RAG")

    if not client.collection_exists("CPU RAG"):
        logfire.info('vector db not found...')

        vector_db()
        return
    logfire.info('vector db found....')
    return





asyncio.run(initialize_vectorstore())