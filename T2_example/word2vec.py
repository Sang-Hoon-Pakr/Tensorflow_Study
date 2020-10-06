import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load Pretrained Word2Vec
embed = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")

words = ["coffee", "cafe", "football", "soccer"]
# Compute embeddings.
embeddings = embed(words)

print(embeddings.shape)

print(embeddings[0])

# Compute similarity matrix. Higher score indicates greater similarity.
for i in range(len(words)):
    for j in range(i,len(words)):
        print("(",words[i], ",", words[j],")",np.inner(embeddings[i], embeddings[j]))