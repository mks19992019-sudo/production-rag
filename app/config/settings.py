from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY:str

    QDRANT_API_KEY:str
    QDRANT_CLUSTER_URL:str
    JINA_API_KEY : str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()



