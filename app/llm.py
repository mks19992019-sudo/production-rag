from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

GROQ_API = os.getenv("GROQ_API_KEY")





def embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )



def model():
    return ChatGroq(mode='llama-3.3-70b-versatile'
                    ,api_key=GROQ_API)