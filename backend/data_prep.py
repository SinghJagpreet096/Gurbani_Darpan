import requests
from tqdm import tqdm
import pandas as pd
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlite3
from backend.config import Config

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

def setup_database():
    """Create SQLite database with proper table structure and primary key"""
    conn = sqlite3.connect(f"{OUTPUT_DIR}mydata.sqlite")
    cursor = conn.cursor()
    
    # Create table with verseId as primary key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_data (
        verseId TEXT PRIMARY KEY,
        ang INTEGER,
        pageNo INTEGER,
        lineNo INTEGER,
        writer TEXT,
        raag TEXT,
        shabadId TEXT,
        verse TEXT,
        translation_en TEXT,
        translation_pu TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database setup completed with verseId as primary key")

def save_chunk(chunk_data):
    """Save a chunk of data to SQLite, ignoring duplicates"""
    if not chunk_data:
        return
    
    df = pd.DataFrame(chunk_data)
    
    # Connect to SQLite database
    conn = sqlite3.connect(f"{OUTPUT_DIR}mydata.sqlite")
    
    # Use INSERT OR IGNORE to avoid duplicates based on primary key
    for _, row in df.iterrows():
        try:
            conn.execute("""
            INSERT OR IGNORE INTO api_data 
            (verseId, ang, pageNo, lineNo, writer, raag, shabadId, verse, translation_en, translation_pu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['verseId'], row['ang'], row['pageNo'], row['lineNo'], 
                row['writer'], row['raag'], row['shabadId'], row['verse'], 
                row['translation_en'], row['translation_pu']
            ))
        except sqlite3.IntegrityError:
            # This should not happen with INSERT OR IGNORE, but just in case
            print(f"Duplicate verseId found: {row['verseId']}")
            continue
    
    conn.commit()
    conn.close()
    print(f"✅ Saved chunk with {len(chunk_data)} verses to SQLite")

# Main processing loop
def main():
    # Setup database with proper table structure
    setup_database()
    
    current_chunk = []
    processed_verse_ids = set()  # Track processed verse IDs to avoid duplicates in memory

    for ang_no in tqdm(range(1, MAX_ANGS + 1), desc="Processing Angs"):
        verses = process_ang(ang_no)
        
        # Filter out duplicates within the current chunk
        for verse in verses:
            verse_id = verse['verseId']
            if verse_id not in processed_verse_ids:
                current_chunk.append(verse)
                processed_verse_ids.add(verse_id)
        
        # Save chunk when reaching CHUNK_SIZE
        if len(current_chunk) >= CHUNK_SIZE or ang_no == MAX_ANGS:
            save_chunk(current_chunk)
            current_chunk = []  # Reset chunk
        
        # Be gentle with the API
        time.sleep(0.15)

if __name__ == "__main__":
    main()