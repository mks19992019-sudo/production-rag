from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.embeddings import JinaEmbeddings



load_dotenv()


GROQ_API = None
JINA_API_KEY = None
_embeddings = None
_jina_embeddings = None

# hugging face Embeddings model
def embedding_model():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    return _embeddings


## jina Embeddings model 
def jinaEmbedding_model():
   
    global _jina_embeddings ,  JINA_API_KEY

    if _jina_embeddings is None:
        if JINA_API_KEY is None:
            JINA_API_KEY = os.getenv("JINA_API_KEY")

        _jina_embeddings = JinaEmbeddings(
            jina_api_key=JINA_API_KEY,
            model_name='jina-embeddings-v4'
        )
    return _jina_embeddings


def model():
    if GROQ_API is None:
        GROQ_API = os.getenv('GROQ_API')


    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API
    )




