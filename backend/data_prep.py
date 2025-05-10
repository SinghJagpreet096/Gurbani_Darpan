import requests
from tqdm import tqdm
import pandas as pd
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlite3
from config import Config


cnf = Config()
# Configuration
BASE_URL = cnf.api_url  # API URL
SOURCE_ID = cnf.source_id  # Source ID for Guru Granth Sahib Ji
CHUNK_SIZE = cnf.chunk_size  # Process 50 Angs at a time
MAX_ANGS = cnf.max_angs # Total Angs in SGGS Ji
OUTPUT_DIR = cnf.database_dir  # Directory to save SQLite database
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
    # Step 3: Save to SQLite
    conn = sqlite3.connect(f"{OUTPUT_DIR}mydata.sqlite")
    df.to_sql("api_data", conn, if_exists="append", index=False)
    conn.close()
    print("✅ Data saved to SQLite (data/mydata.sqlite)")
   

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

