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
        # self.embeddings_path = EMBEDDINGS_PATH
        self.dataset: pd.DataFrame = pd.DataFrame()
        

    def load_dataset(self, dataset_path: str, sample_size: int = 100, issample: bool = False):
        if issample:
            dataset = pd.read_parquet(dataset_path).sample(sample_size, random_state=RANDOM_STATE)
            dataset = dataset.reset_index(drop=True)
            print(f"Created new sample of {len(dataset)} verses")
            return dataset
        else:
            dataset =pd.read_parquet(dataset_path)
            print(f"Loaded {len(dataset)} pre-embedded verses")
            return dataset
    
    def load_embeddings(self, embeddings_path: str):
        dataset = pd.read_parquet(embeddings_path)
        print(f"Loaded {len(self.dataset)} pre-embedded verses from sample")
        return dataset


    def is_embedding_exist(self, dataset_path: str, embeddings_path: str, issample: bool = False):
        if os.path.exists(embeddings_path):
            # Load existing sample with embeddings
            return True
        else:
            return False
            

    def generate_embeddings(self, dataset_path: str, embeddings_path: str, issample: bool = False):
        
        if self.is_embedding_exist(dataset_path, embeddings_path, issample):
            self.dataset = self.load_embeddings(embeddings_path)
        else:
            self.dataset = self.load_dataset(dataset_path, SAMPLE_SIZE, issample)

        if 'embeddings' not in self.dataset.columns:
            print("Generating fresh embeddings...")
            embeddings = []
            for shabad in tqdm(self.dataset['shabad_text'], desc="Processing"):
                embeddings.append(self.model.encode(shabad))
            self.dataset['embeddings'] = embeddings
            self.dataset.to_parquet(embeddings_path)
            print(f"Saved embeddings to {embeddings_path}")
        embeddings_array = np.stack(list(self.dataset['embeddings'].values))
        return self.dataset, embeddings_array
    
    

   
        
if __name__ == "__main__":
    embedding = Embedding()
    # embedding.is_embedding_exist()
    dataset_path = "backend/db/merged_shabad.parquet"
    embeddings_path = "backend/db/shabad_embeddings.parquet"
    d, e = embedding.generate_embeddings(dataset_path, embeddings_path, issample=True)
    print(f"Dataset shape: {d.shape}")
    print(f"Embeddings shape: {e.shape}")
    # Example usage
    # query = "What does Gurbani say about ego?"
    # response = embedding.search(query, top_k=2)
    # print(response)

    
