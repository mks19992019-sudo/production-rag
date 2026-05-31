from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API = os.getenv("GROQ_API_KEY")

_embeddings = None


def embedding_model():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    return _embeddings


def model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API
    )


