from langchain_huggingface import HuggingFaceEmbeddings

def embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
