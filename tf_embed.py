import tensorflow_hub as hub
import ssl
import keras

ssl._create_default_https_context = ssl._create_unverified_context

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed(
    [
        "sáng hôm qua tôi không đi học",
        "tôi sẽ lên công ty vào chiều nay",
    ]
)

print(embeddings)
cosine_similarity = keras.metrics.CosineSimilarity()(embeddings[0], embeddings[1])
print(cosine_similarity)
