from src.llm_gateway import jinaEmbedding_model

model = jinaEmbedding_model()

print(model)
print("Jina embedding model loaded successfully")

document = model.embed_documents([
    "My name is Mohit",
    "I live in India",
    "I am learning Deep Learning"
])

q= model.embed_query('what is my name')


print(len(q))

print(len(document[0]))

from sklearn.metrics.pairwise import cosine_similarity

score = cosine_similarity([q],document)[0]
print(score)





