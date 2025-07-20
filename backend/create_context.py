from generateEmbeddings import Embedding, chromaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from config import Config
cnf = Config()
import sqlite3
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
        
    def get_verseid(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=top_k  # how many results to return
        )
        return results['ids']
    
    def get_shabad(self, verse_ids: list):
        verse_ids = tuple(map(int, verse_ids))
        print(f"verse_ids: {verse_ids}")
        with sqlite3.connect(Config.database_path) as conn:
                query = f"""SELECT verse, shabadId, writer, ang from api_data
                            WHERE shabadId IN (SELECT DISTINCT shabadId FROM api_data WHERE verseId in {verse_ids})
                            order by verseId
                """
                df = pd.read_sql_query(query, conn)
        contexts = []
        for shabadid in df['shabadId'].unique():
            context = {}
            # context['shabad'] = ""
            # context['writer'] = ""
            # context['ang'] = ""
            shabad = df['verse'][df['shabadId'] == shabadid].tolist()
            writer = df['writer'][df['shabadId'] == shabadid].iloc[0]
            ang = df['ang'][df['shabadId'] == shabadid].iloc[0]

            shabad = " ".join(shabad)
            context['shabad'] = shabad
            context['writer'] = writer
            context['ang'] = ang
            contexts.append(context)
        return contexts


    def provide_context(self, query: str, top_k: int = 3):
        verse_ids = self.get_verseid(query, top_k)
        # shabads = []
        # for i in verse_ids[0]:
        #     shabad = self.get_shabad(i)
        #     shabads.append(shabad)
        shabads = self.get_shabad(verse_ids[0])
        return shabads


  
if __name__ == "__main__":
    context = contextChroma()
    query = "What does Gurbani say about ego?"
    response = context.provide_context(query)
    for i in response:
        print(i)
        print("-" * 50)