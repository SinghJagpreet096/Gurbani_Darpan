import requests
from tqdm import tqdm
import pandas as pd
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
BASE_URL = "https://api.banidb.com/v2/angs"
SOURCE_ID = 'G'  # Gurmukhi source
CHUNK_SIZE = 50  # Process 50 Angs at a time
MAX_ANGS = 1430  # Total Angs in SGGS Ji
OUTPUT_DIR = "gurbani_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup retry mechanism
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

def process_ang(ang_no):
    """Process a single Ang and return its verses"""
    try:
        response = session.get(f"{BASE_URL}/{ang_no}/{SOURCE_ID}", timeout=10)
        response.raise_for_status()
        ang = response.json()
        
        verses = []
        for verse in ang.get('page', []):
            verses.append({
                'ang': ang_no,
                'pageNo': verse.get('pageNo'),
                'lineNo': verse.get('lineNo'),
                'writer': verse.get('writer', {}).get('english', 'Unknown'),
                'raag': verse.get('raag', {}).get('english', 'Unknown'),
                'verseId': verse.get('verseId'),
                'shabadId': verse.get('shabadId'),
                'verse': verse.get('verse', {}).get('unicode', ''),
                'translation_en': verse.get('translation', {}).get('en', {}).get('bdb', ''),
                'translation_pu': verse.get('translation', {}).get('pu', {}).get('bdb', {}).get('unicode', '')
            })
        return verses
    
    except Exception as e:
        print(f"\nError processing Ang {ang_no}: {str(e)[:100]}")
        return []

def save_chunk(chunk_data, chunk_id):
    """Save a chunk of data to Parquet"""
    if not chunk_data:
        return
    
    df = pd.DataFrame(chunk_data)
    output_path = f"{OUTPUT_DIR}/chunk_{chunk_id:03d}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} verses to {output_path}")

# Main processing loop
current_chunk = []
chunk_id = 0

for ang_no in tqdm(range(1, MAX_ANGS + 1), desc="Processing Angs"):
    verses = process_ang(ang_no)
    current_chunk.extend(verses)
    
    # Save chunk when reaching CHUNK_SIZE
    if ang_no % CHUNK_SIZE == 0 or ang_no == MAX_ANGS:
        save_chunk(current_chunk, chunk_id)
        current_chunk = []
        chunk_id += 1
    
    # Be gentle with the API
    time.sleep(0.15)

# Merge all chunks (optional)
def merge_chunks():
    """Combine all chunk files into one dataset"""
    chunk_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('chunk_')]
    df = pd.concat(
        (pd.read_parquet(f"{OUTPUT_DIR}/{f}") for f in chunk_files),
        ignore_index=True
    )
    df.to_parquet("merged_gurbani.parquet", index=False)
    print(f"\nMerged {len(df)} total verses")

merge_chunks()