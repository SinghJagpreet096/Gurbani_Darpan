from generateEmbeddings import Embedding, chromaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from config import Config
cnf = Config()
# Config
MODEL_PATH = 'paraphrase-multilingual-mpnet-base-v2'
MAIN_DATA_PATH = cnf.database_dir + "merged_shabad.parquet"
EMBEDDINGS_PATH = cnf.database_dir + "shabad_embeddings.parquet"
# Load or Create Sample

class Context(Embedding):
    def __init__(self, model_path: str = 'paraphrase-multilingual-mpnet-base-v2'):
        super().__init__(model_path)
        self.dataset, _ = self.generate_embeddings(
            dataset_path="backend/db/merged_shabad.parquet",
            embeddings_path="backend/db/shabad_embeddings.parquet",
            issample=False
        )

        
      
    def provide_context(self, query: str, top_k: int = 3):
            # Embed the query
            query_embed = self.model.encode(query)
            query_embed = np.array(query_embed).reshape(1, -1)
            
            # Calculate similarities
            sim_scores  = cosine_similarity(
                query_embed,
                np.stack(self.dataset['embeddings'].tolist())
            )[0]
            
            # Get top matches
            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            results = self.dataset.iloc[top_indices]
            
            # Format response
            metadata = {}
            response = f"Query: {query}\n\n"
            for _, row in results.iterrows():
                response += f"""
                Shabad Text: {row['shabad_text']},
                Verse: {row['verse']},
                Translation_english: {row['translation_en']},
                Translation_punjabi: {row['translation_pu']}
                Source: Ang {row['ang']},
                Writer: {row['writer']},
                {'-'*50}
                """
                metadata = {
                    "ang": row['ang'],
                    # "pageNo": row['pageNo'],
                    # "lineNo": row['lineNo'],
                    "writer": row['writer'],
                    "raag": row['raag'],
                    "verseId": row['verseId'],
                    "shabadId": row['shabadId'],
                    "verse": row['verse'],
                    "translation_en": row['translation_en'],
                    "translation_pu": row['translation_pu']
                }
            
            return response, metadata

class contextChroma(chromaEmbedding):
    def __init__(self):
        super().__init__()
        
    

    def provide_context(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query], # Chroma will embed this for you
            n_results=top_k # how many results to return
        )
        return results


  
if __name__ == "__main__":
    context = contextChroma()
    query = "What does Gurbani say about ego?"
    response = context.provide_context(query)
    print(response)
    # print(f"{'-'*50}\n metadata:{results}") # Get verses from previous function