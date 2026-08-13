from langchain_community.document_loaders import WebBaseLoader , TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm_gateway import jinaEmbedding_model
from src.config.settings import settings

from qdrant_client import QdrantClient
import asyncio


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
        chunk_size = 800,
        chunk_overlap = 100,
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
    docs = extract()

    chunks = chunking_Recursive_text_split(docs)
    

    vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=jinaEmbedding_model(),
            url = settings.QDRANT_CLUSTER_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name= "cpu_docs",
            

        )

    return vector_store



# internally

#info = client.get_collection("cpu_docs")
#print(info.points_count)

async def initialize_vectorstore():
    if not client.collection_exists("cpu_docs"):
        vector_db()
        return
    return




if __name__ =="__main__":
    asyncio.run(initialize_vectorstore())