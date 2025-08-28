from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import sqlite3
import chromadb
from chromadb.config import Settings
from backend.config import Config
import chromadb.utils.embedding_functions as embedding_functions
from backend.embedding_function import MyEmbeddingFunction

load_dotenv()



# Config
SAMPLE_SIZE = 100
RANDOM_STATE = 42
MAIN_DATA_PATH = Config().database_path
EMBEDDINGS_PATH = Config().embeddings_path
MODEL_PATH = 'paraphrase-multilingual-mpnet-base-v2'
chroma_key = os.getenv("chroma_token")


# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key=os.getenv("HF_TOKEN_R"),
#     model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# )




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
    

     
class chromaEmbedding():
    def __init__(self, model_path: str = MODEL_PATH):
        # super().__init__(model_path)
        self.model = SentenceTransformer(model_path)
        # Initialize with low-memory settings
        self.client = chromadb.HttpClient(
            ssl=True,
            host="https://api.trychroma.com:8000",
            tenant='61c02c3f-fe20-493e-bfb4-d5c2efe25b2e',
            database='gurbani_db',
            headers={
                'x-chroma-token': chroma_key
            }
        )
        self.collection = self.client.get_or_create_collection(
            name='paraphrase-multilingual-mpnet-base-v2',
            embedding_function=MyEmbeddingFunction()
        )


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

    def is_embedding_exist(self, embeddings_path: str):
        return os.path.exists(embeddings_path)
    
    def load_embeddings(self, embeddings_path: str):
        pass

    def generate_embeddings(self, dataset_path: str, embeddings_path: str, issample: bool = False, sample_size: int = 100):

        if issample:
            self.dataset = self.load_dataset(dataset_path, sample_size, issample)
        else:
            self.dataset = self.load_dataset(dataset_path, sample_size, issample)

        # Batch processing for low RAM
        # df = self.load_dataset(dataset_path, SAMPLE_SIZE, issample)
        df = self.dataset
        batch_size = 100  # Adjust based on available RAM
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch = df.iloc[i:i+batch_size]
            self.collection.upsert(
                documents=batch['verse'].tolist(),
                metadatas=batch[['ang', 'raag', 'shabadId']].to_dict('records'),
                # embeddings=self.model.encode(batch['verse'].tolist(), show_progress_bar=False).tolist(),
                ids=[str(i) for i in batch['verseId']]
            )
            # print(f"Processed batch {i//batch_size + 1}/{(len(df) + batch_size - 1) // batch_size} with {len(batch)} verses}")

        # self.client.persist()
        self.collection.peek()
        print(f"Stored {len(df)} verses")

def main():
   # ample usage
    embedding = chromaEmbedding()
    # dataset_path = "backend/data/mydata.sqlite"
    dataset_path = Config().database_path
    d = embedding.load_dataset(dataset_path=dataset_path, sample_size=200)
    print(d.head())
    embedding.generate_embeddings(dataset_path=dataset_path,
                                   embeddings_path=Config().embeddings_path, 
                                   issample=True, sample_size=10000)
if __name__ == "__main__":
    main()

    
