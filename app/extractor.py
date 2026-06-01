from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from .llm import embedding_model
import os
from  dotenv import load_dotenv


load_dotenv()

from qdrant_client import QdrantClient

#qdrant_url = os.getenv("QDRANT_URL")

# here we cant use the try and excpet becuse it give some error during the 
# docker creation so i remove. the claude api key know from the env and just use the one servies 

"""if qdrant_url:
    client = QdrantClient(
        url=qdrant_url,
        api_key=os.getenv("QDRANT_API_KEY")
    )
else:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL_LOCAL")
    )"""


client = QdrantClient(
        url=os.getenv("QDRANT_URL_LOCAL")
)
def extract(url):
    loader = WebBaseLoader(url)
    return loader.load()


def chunking(documents):
    chunker = SemanticChunker(
        embeddings=embedding_model(),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=10,
        min_chunk_size=50,
    )

    return chunker.split_documents(documents)


def vector_db(url):
    docs = extract(url)

    chunks = chunking(docs)
    

    # from_documents crete the colletion file if not exit 
    """try:
        vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model(),
        url=os.getenv("QDRANT_URL"),
        collection_name="cpu_docs",
        api_key = os.getenv("QDRANT_API_KEY")
    )"""
    #except:

    vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model(),
            url = os.getenv('QDRANT_URL_LOCAL'),
            collection_name= "cpu_docs"

        )

    return vector_store


# to build the vectore data base




#info = client.get_collection("cpu_docs")

#print(info.points_count)
async def initialize_vectorstore():
    if not client.collection_exists("cpu_docs"):
        vector_db("https://cpur.in")
        return
    return


