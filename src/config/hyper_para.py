

# production 'parameters'
class production:
    top_k: int = 6
    rerank_top_k: int = 6
    chunk_size: int = 250
    chunk_overlap: int = 50


## eval parameter
class experiment:
    top_k: int = 6
    rerank_top_k: int = 6
    chunk_size: int = 250
    chunk_overlap: int = 50



load_para = experiment()

