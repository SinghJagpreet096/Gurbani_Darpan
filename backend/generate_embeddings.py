import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import sqlite3
import chromadb
from chromadb.config import Settings

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
    
    
class chromaEmbedding(Embedding):
    def __init__(self, model_path: str = MODEL_PATH):
        super().__init__(model_path)
        self.model = SentenceTransformer(model_path, device='cpu')
        # Initialize with low-memory settings
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                                persist_directory="./chroma_db",
                                                anonymized_telemetry=False
                            ))


    def load_dataset(self, dataset_path: str, sample_size: int = 100, issample: bool = False):
        conn = sqlite3.connect(dataset_path)
        if issample:
            query = f"SELECT * FROM api_data ORDER BY RANDOM() LIMIT {sample_size}"
            dataset = pd.read_sql_query(query, conn)
            dataset = dataset.reset_index(drop=True)
            print(f"Created new sample of {len(dataset)} verses")
        else:
            query = "SELECT * FROM api_data"
            dataset = pd.read_sql_query(query, conn)
            print(f"Loaded {len(dataset)} pre-embedded verses")
        conn.close()
        return dataset
    
    def is_embedding_exist(self, dataset_path: str, embeddings_path: str, issample: bool = False):
        return os.path.exists(embeddings_path)
    
    def load_embeddings(self, embeddings_path: str):
        pass

    def generate_embeddings(self, dataset_path: str, embeddings_path: str, issample: bool = False):    
        if self.is_embedding_exist(dataset_path, embeddings_path, issample):
            print("embedding already exists")
            return
        else:
            self.dataset = self.load_dataset(dataset_path, SAMPLE_SIZE, issample)

        collection = self.client.create_collection(
            name="gurbani",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.model.encode
        )
        
        # Batch processing for low RAM
        df = self.load_dataset(dataset_path, SAMPLE_SIZE, issample)
        batch_size = 100  # Adjust based on available RAM
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            collection.add(
                documents=batch['verse'].tolist(),
                metadatas=batch[['ang', 'raag', 'shabadId']].to_dict('records'),
                ids=[f"id_{x}" for x in range(i, i+len(batch))]
            )

        self.client.persist()
        print(f"Stored {len(df)} verses")

if __name__ == "__main__":
    # Example usage
    embedding = chromaEmbedding()
    dataset_path = "backend/data/mydata.sqlite"
    d = embedding.load_dataset(dataset_path=dataset_path)
    print(d.head())
    embedding.generate_embeddings(dataset_path=dataset_path, embeddings_path=EMBEDDINGS_PATH, issample=True)

    
