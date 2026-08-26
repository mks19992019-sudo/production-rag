from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings import JinaEmbeddings
from src.config.settings import settings







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
            JINA_API_KEY = settings.JINA_API_KEY

        _jina_embeddings = JinaEmbeddings(
            jina_api_key=JINA_API_KEY,
            model_name='jina-embeddings-v4'
        )
    return _jina_embeddings


def model():
    global GROQ_API
    if GROQ_API is None:
        GROQ_API = settings.GROQ_API_KEY


    return ChatGroq(
        model="groq/compound",
        api_key=GROQ_API
    )



# we need deepEval model wraper

from deepeval.models import DeepEvalBaseLLM

class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.llm = model()

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "llama-3.3-70b-versatile"

