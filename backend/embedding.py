import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# Config
SAMPLE_SIZE = 100
RANDOM_STATE = 42
MAIN_DATA_PATH = "backend/db/merged_gurbani.parquet"
EMBEDDINGS_PATH = "backend/db/gurbani_embeddings.parquet"
MODEL_PATH = 'paraphrase-multilingual-mpnet-base-v2'

# 1. Load or Create Sample
class Embedding:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = SentenceTransformer(model_path)
        self.embeddings_path = EMBEDDINGS_PATH
        self.dataset = None
        

    def load_dataset(self, dataset_path: str = MAIN_DATA_PATH, sample_size: int = 100, issample: bool = False):
        if issample:
            self.dataset = pd.read_parquet(dataset_path).sample(sample_size, random_state=RANDOM_STATE)
            self.dataset = self.dataset.reset_index(drop=True)
            print(f"Created new sample of {len(self.dataset)} verses")
        else:
            self.dataset = pd.read_parquet(dataset_path)
            print(f"Loaded {len(self.dataset)} pre-embedded verses")

    def is_embedding_exist(self):
        if os.path.exists(self.embeddings_path):
            # Load existing sample with embeddings
            self.dataset = pd.read_parquet(self.embeddings_path)
            print(f"Loaded {len(self.dataset)} pre-embedded verses from sample")
        else:
            self.load_dataset(MAIN_DATA_PATH, SAMPLE_SIZE, issample=False)

    def generate_embeddings(self):
        self.is_embedding_exist()
        if 'embeddings' not in self.dataset.columns:
            print("Generating fresh embeddings...")
            embeddings = []
            for verse in tqdm(self.dataset['verse'], desc="Processing"):
                embeddings.append(self.model.encode(verse))
            self.dataset['embeddings'] = embeddings
            self.dataset.to_parquet(self.embeddings_path)
            print(f"Saved embeddings to {self.embeddings_path}")
        embeddings_array = np.stack(self.dataset['embeddings'].values)
    
    def top_k_verse(self, query: str, top_k: int = 3):
        # Embed the query
        query_embed = self.model.encode(query)
        
        # Calculate similarities
        sim_scores = cosine_similarity(
            [query_embed],
            np.stack(self.dataset['embeddings'])
        )[0]
        
        # Get top matches
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        results = self.dataset.iloc[top_indices]
        
        # Format response
        response = f"Query: {query}\n\n"
        for _, row in results.iterrows():
            response += f"""
            Verse: {row['verse']},
            Verse: {row['verseId']},
            Translation_en: {row['translation_en']},
            Translation_pu: {row['translation_pu']}
            Source: Ang {row['ang']},
            Shabad: {row['shabadId']},
            Similarity: {sim_scores[_]:.2f}
            {'-'*50}
            """
        return response
        
        
if __name__ == "__main__":
    embedding = Embedding()
    # embedding.is_embedding_exist()
    embedding.generate_embeddings()
    # Example usage
    query = "What does Gurbani say about ego?"
    response = embedding.top_k_verse(query, top_k=2)
    print(response)
    # print(res)
# 5. Example Usage
# print(answer_gurbani("What does Gurbani say about ego?"))
# print(answer_gurbani("How to find peace?", top_k=2))