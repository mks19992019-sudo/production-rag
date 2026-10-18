from pydantic_settings import BaseSettings , SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    GROQ_API_KEY1:str
    GROQ_API_KEY2:str
    model_ : str


    QDRANT_API_KEY:str
    QDRANT_CLUSTER_URL:str
    JINA_API_KEY : str

    model_config = SettingsConfigDict(env_file=".env")

    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str 


settings = Settings()



