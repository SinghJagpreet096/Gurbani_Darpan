import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 1. Load Data
df = pd.read_parquet("backend/merged_gurbani.parquet")  # Your dataset
print(f"Loaded {len(df)} verses")

# df = df.sample(100, random_state=42)
# df.reset_index(inplace=True)  # For testing, use a smaller sample
# 2. Initialize Embedding Model (multilingual works best)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 3. Precompute Embeddings (do this once and save)
if 'embeddings' not in df.columns:
    print("Generating embeddings...")
    # tqdm.pandas(desc="Generating embeddings")
    df['embeddings'] = df['verse'].apply(lambda x: model.encode(x))
    df.to_parquet("backend/gurbani_embeddings.parquet")  # Save for future

# 4. RAG Query Function
def answer_gurbani(query, top_k=3):
    # Embed the query
    query_embed = model.encode(query)
    
    # Calculate similarities
    sim_scores = cosine_similarity(
        [query_embed],
        np.stack(df['embeddings'])
    )[0]
    
    # Get top matches
    top_indices = np.argsort(sim_scores)[-top_k:][::-1]
    results = df.iloc[top_indices]
    
    # Format response
    response = f"Query: {query}\n\n"
    for _, row in results.iterrows():
        response += f"""
        Verse: {row['verse']}
        Translation: {row['translation_en']}
        Source: Ang {row['ang']}
        Similarity: {sim_scores[_]:.2f}
        {'-'*50}
        """
    return response

# 5. Example Usage
print(answer_gurbani("What does Gurbani say about ego?"))
print(answer_gurbani("How to find peace?", top_k=2))