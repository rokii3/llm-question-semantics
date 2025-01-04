from sentence_transformers import SentenceTransformer # loads the model and generates embeddings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# first check a few examples in detail
print("Checking some example pairs:\n")

# look at the pair with highest similarity
max_sim_idx = data['cosine_similarity'].idxmax()
print("Highest similarity pair:")
print(f"Source: {data.loc[max_sim_idx, 'source']}")
print(f"Target: {data.loc[max_sim_idx, 'target']}")
print(f"Similarity: {data.loc[max_sim_idx, 'cosine_similarity']:.3f}")
print(f"Language: {data.loc[max_sim_idx, 'language']}\n")

# look at the pair with lowest similarity
min_sim_idx = data['cosine_similarity'].idxmin()
print("Lowest similarity pair:")
print(f"Source: {data.loc[min_sim_idx, 'source']}")
print(f"Target: {data.loc[min_sim_idx, 'target']}")
print(f"Similarity: {data.loc[min_sim_idx, 'cosine_similarity']:.3f}")
print(f"Language: {data.loc[min_sim_idx, 'language']}\n")

# let's also check embedding dimensions and values
print("Checking embedding properties:")
source_emb = data.loc[0, 'source_embedding']  # look at first embedding
print(f"Embedding dimensions: {len(source_emb)}")
print(f"Sample values from first embedding: {source_emb[:5]}")  # first 5 values

# verify our cosine similarity calculation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# recalculate similarity for a few pairs using sklearn
print("\nDouble-checking similarities with sklearn:")
for i in range(3):  # check first 3 pairs
    src_emb = np.array(data.loc[i, 'source_embedding']).reshape(1, -1)
    tgt_emb = np.array(data.loc[i, 'target_embedding']).reshape(1, -1)
    sklearn_sim = cosine_similarity(src_emb, tgt_emb)[0][0]
    our_sim = data.loc[i, 'cosine_similarity']
    print(f"Pair {i}:")
    print(f"Our calculation: {our_sim:.3f}")
    print(f"Sklearn calculation: {sklearn_sim:.3f}")