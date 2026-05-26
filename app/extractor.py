from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from llm import embedding_model

def extract(url):
    loader = WebBaseLoader(url)
    return loader.load()


def chunking(documents):
    chunker = SemanticChunker(
        embeddings=embedding_model(),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=200,
    )

    return chunker.split_documents(documents)


def vector_db(url):
    docs = extract(url)

    chunks = chunking(docs)
    

    # from_documents crete the colletion file if not exit 
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model(),
        url="http://localhost:6333",
        collection_name="cpu_docs",
    )

    return vector_store


if __name__ == "__main__":
    vector_db("https://cpur.in")